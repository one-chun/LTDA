import sys
import os
import argparse
import torch
import torch.nn as nn
import math
import json
import random
import time
import numpy as np
sys.path.append('src')
import Models
import ppi_data
import utils

def str2bool(v):
	"""
	Converts string to bool type; enables command line 
	arguments in the format of '--arg1 true --arg2 false'
	"""
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	return


def str2intlist(v):
	if isinstance(v, list):
		return [int(item) for item in v]
	if isinstance(v, str):
		items = [item.strip() for item in v.split(',') if item.strip()]
		if not items:
			raise argparse.ArgumentTypeError('At least one diffusion scale is required.')
		return [int(item) for item in items]
	raise argparse.ArgumentTypeError('Comma-separated integer list expected.')


def str2floatlist(v):
	if v is None:
		return None
	if isinstance(v, list):
		return [float(item) for item in v]
	if isinstance(v, str):
		items = [item.strip() for item in v.split(',') if item.strip()]
		if not items:
			return None
		return [float(item) for item in items]
	raise argparse.ArgumentTypeError('Comma-separated float list expected.')


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)


def resolve_bce_pos_weight(data, args, device):
	mode = getattr(args, 'bce_pos_weight_mode', 'none')
	if mode == 'none':
		return None

	if mode == 'manual':
		values = getattr(args, 'bce_pos_weight_values', None)
		if values is None:
			raise ValueError('bce_pos_weight_values is required when bce_pos_weight_mode=manual')
		if len(values) != data.edge_attr.shape[1]:
			raise ValueError(
				f'bce_pos_weight_values must have {data.edge_attr.shape[1]} entries, got {len(values)}'
			)
		weights = torch.tensor(values, dtype=torch.float, device=device)
	else:
		train_attr = data.edge_attr[data.train_mask].float()
		num_train_edges = float(train_attr.shape[0])
		pos = train_attr.sum(dim=0)
		neg = num_train_edges - pos
		weights = neg / pos.clamp_min(1.0)

	clip_value = float(getattr(args, 'bce_pos_weight_clip', 0.0))
	if clip_value > 0.0:
		weights = torch.clamp(weights, max=clip_value)

	return weights.to(device)


def resolve_threshold_grid(args):
	grid = getattr(args, 'eval_threshold_grid', None)
	if grid is None or len(grid) == 0:
		return [round(v, 2) for v in np.arange(0.05, 1.0, 0.05)]
	return [float(v) for v in grid]


def collect_ltda_debug_stats(model, node_degree=None):
	base_model = getattr(model, 'module', model)
	deg = None
	if node_degree is not None:
		deg = torch.log1p(node_degree.detach().float().cpu()).unsqueeze(-1)

	stats = []
	for name, module in base_model.named_modules():
		if not isinstance(module, Models.CurvatureHyperbolicDiffusionKernel):
			continue
		alpha_logit = module.alpha_logit.detach().cpu()
		deg_slope = torch.nn.functional.softplus(module.deg_slope_raw.detach().cpu())
		scale_weights = torch.softmax(module.scale_weights.detach().cpu(), dim=0)
		entry = {
			'name': name,
			'scales': list(module.scales),
			'use_self_loop': bool(getattr(module, 'use_self_loop', True)),
			'return_manifold': bool(getattr(module, 'return_manifold', True)),
			'alpha_deg0': float(torch.sigmoid(alpha_logit).item()),
			'deg_slope': float(deg_slope.item()),
			'scale_weights': [float(weight) for weight in scale_weights.tolist()],
		}
		if deg is not None:
			alpha_nodes = torch.sigmoid(alpha_logit - deg_slope * deg)
			entry['alpha_mean'] = float(alpha_nodes.mean().item())
			entry['alpha_min'] = float(alpha_nodes.min().item())
			entry['alpha_max'] = float(alpha_nodes.max().item())
		stats.append(entry)
	return stats


def summarize_ltda_debug_stats(stats):
	if len(stats) == 0:
		return None, []

	alpha_deg0_mean = sum(item['alpha_deg0'] for item in stats) / len(stats)
	deg_slope_mean = sum(item['deg_slope'] for item in stats) / len(stats)
	weight_by_scale = {}
	count_by_scale = {}
	alpha_mean_values = []
	for item in stats:
		if 'alpha_mean' in item:
			alpha_mean_values.append(item['alpha_mean'])
		for scale, weight in zip(item['scales'], item['scale_weights']):
			weight_by_scale[scale] = weight_by_scale.get(scale, 0.0) + weight
			count_by_scale[scale] = count_by_scale.get(scale, 0) + 1

	scale_summary = ",".join(
		f"{scale}:{weight_by_scale[scale] / count_by_scale[scale]:.3f}"
		for scale in sorted(weight_by_scale.keys())
	)
	use_self_loop = all(item.get('use_self_loop', True) for item in stats)
	return_manifold = all(item.get('return_manifold', True) for item in stats)
	summary = f"LTDA self_loop {use_self_loop}, return_manifold {return_manifold}, alpha0_mean {alpha_deg0_mean:.4f}, deg_slope_mean {deg_slope_mean:.4f}, scale_weight_mean [{scale_summary}]"
	if len(alpha_mean_values) > 0:
		alpha_mean = sum(alpha_mean_values) / len(alpha_mean_values)
		summary = f"{summary}, alpha_train_mean {alpha_mean:.4f}"

	detail_lines = []
	for item in stats:
		scale_detail = ",".join(
			f"{scale}:{weight:.4f}"
			for scale, weight in zip(item['scales'], item['scale_weights'])
		)
		line = f"{item['name']} self_loop {item.get('use_self_loop', True)}, return_manifold {item.get('return_manifold', True)}, alpha0 {item['alpha_deg0']:.4f}, deg_slope {item['deg_slope']:.4f}, scale_weights [{scale_detail}]"
		if 'alpha_mean' in item:
			line += f", alpha_train_mean {item['alpha_mean']:.4f}, alpha_train_range [{item['alpha_min']:.4f},{item['alpha_max']:.4f}]"
		detail_lines.append(line)
	return summary, detail_lines


def apply_thresholds(y_score, thresholds):
	if np.isscalar(thresholds):
		return (y_score >= float(thresholds)).astype(int)
	thresholds = np.asarray(thresholds, dtype=float).reshape(1, -1)
	return (y_score >= thresholds).astype(int)


def compute_multilabel_confusion_metrics(y_true, y_pred):
	tp = int(np.logical_and(y_pred == 1, y_true == 1).sum())
	tn = int(np.logical_and(y_pred == 0, y_true == 0).sum())
	fp = int(np.logical_and(y_pred == 1, y_true == 0).sum())
	fn = int(np.logical_and(y_pred == 0, y_true == 1).sum())
	total = y_true.size
	acc = float((tp + tn) / (total + 1e-10))
	precision = float(tp / (tp + fp + 1e-10))
	recall = float(tp / (tp + fn + 1e-10))
	micro_f1 = float(2 * precision * recall / (precision + recall + 1e-10))
	return {
		"acc": acc,
		"precision": precision,
		"recall": recall,
		"micro_f1": micro_f1,
	}


def compute_binary_f1(y_true, y_pred):
	y_true = np.asarray(y_true).reshape(-1)
	y_pred = np.asarray(y_pred).reshape(-1)
	tp = int(np.logical_and(y_pred == 1, y_true == 1).sum())
	fp = int(np.logical_and(y_pred == 1, y_true == 0).sum())
	fn = int(np.logical_and(y_pred == 0, y_true == 1).sum())
	precision = float(tp / (tp + fp + 1e-10))
	recall = float(tp / (tp + fn + 1e-10))
	return float(2 * precision * recall / (precision + recall + 1e-10))


def select_eval_thresholds(y_true, y_score, args):
	mode = getattr(args, 'eval_threshold_mode', 'fixed')
	if mode == 'fixed':
		threshold = float(getattr(args, 'eval_threshold', 0.5))
		return {"mode": mode, "thresholds": threshold, "display": f"{threshold:.4f}"}

	grid = resolve_threshold_grid(args)
	if mode == 'global_search':
		best_threshold = float(grid[0])
		best_f1 = -1.0
		for threshold in grid:
			y_pred = apply_thresholds(y_score, threshold)
			metrics = compute_multilabel_confusion_metrics(y_true, y_pred)
			if metrics["micro_f1"] > best_f1:
				best_f1 = metrics["micro_f1"]
				best_threshold = float(threshold)
		return {"mode": mode, "thresholds": best_threshold, "display": f"{best_threshold:.4f}"}

	if mode == 'per_label_search':
		best_thresholds = []
		for label_idx in range(y_true.shape[1]):
			label_true = y_true[:, label_idx]
			label_score = y_score[:, label_idx]
			label_best_threshold = float(grid[0])
			label_best_f1 = -1.0
			for threshold in grid:
				label_pred = (label_score >= float(threshold)).astype(int)
				label_f1 = compute_binary_f1(label_true, label_pred)
				if label_f1 > label_best_f1:
					label_best_f1 = label_f1
					label_best_threshold = float(threshold)
			best_thresholds.append(label_best_threshold)
		display = ",".join(f"{value:.4f}" for value in best_thresholds)
		return {"mode": mode, "thresholds": best_thresholds, "display": display}

	raise ValueError(f"Unknown eval_threshold_mode: {mode}")


def build_labelwise_bce_losses(output, label, loss_fn):
	pos_weight = getattr(loss_fn, 'pos_weight', None)
	per_entry = nn.functional.binary_cross_entropy_with_logits(
		output,
		label,
		reduction='none',
		pos_weight=pos_weight,
	)
	return [per_entry[:, idx].mean() for idx in range(per_entry.shape[1])]


def _clone_grad_list(grad_list):
	return [grad.clone() for grad in grad_list]


def _grad_dot(grad_list_a, grad_list_b):
	total = None
	for grad_a, grad_b in zip(grad_list_a, grad_list_b):
		value = (grad_a * grad_b).sum()
		total = value if total is None else total + value
	if total is None:
		return torch.tensor(0.0)
	return total


def apply_label_agnostic_gradient_conflict(model, task_losses):
	params = [param for param in model.parameters() if param.requires_grad]
	task_grads = []
	task_num = len(task_losses)

	for idx, task_loss in enumerate(task_losses):
		grads = torch.autograd.grad(
			task_loss,
			params,
			retain_graph=idx < task_num - 1,
			create_graph=False,
			allow_unused=True,
		)
		task_grads.append([
			torch.zeros_like(param) if grad is None else grad.detach()
			for param, grad in zip(params, grads)
		])

	projected_grads = [_clone_grad_list(grad_list) for grad_list in task_grads]
	for i in range(task_num):
		for j in range(task_num):
			if i == j:
				continue
			dot = _grad_dot(projected_grads[i], task_grads[j])
			if dot.item() < 0.0:
				denom = _grad_dot(task_grads[j], task_grads[j]).clamp_min(1e-12)
				coeff = dot / denom
				projected_grads[i] = [
					grad_i - coeff * grad_j
					for grad_i, grad_j in zip(projected_grads[i], task_grads[j])
				]

	for param_idx, param in enumerate(params):
		merged = sum(projected_grads[task_idx][param_idx] for task_idx in range(task_num)) / float(task_num)
		param.grad = merged

#graph: embed1: data, edge1: local , edge2 global edges 
def train(model,data,loss_fn,optimizer,device,result_prefix=None,batch_size=512,epochs=100,scheduler=None,global_best_f1=0.0,args=None):
	with open(result_prefix+'.txt','w') as f:
		f.write('')
	#torch.backends.cudnn.benchmark =  True
	#torch.backends.cudnn.enabled =  True
	print('begin training')
	best_f1,best_epoch = 0.0,0
	result = None
	use_amp = bool(getattr(args, 'amp', False)) and torch.cuda.is_available()
	use_grad_conflict_correction = bool(getattr(args, 'use_grad_conflict_correction', False))
	scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
	val_size = len(data.val_mask)
	aly_data = None
	best_threshold_state = None
	for epoch in range(epochs):
		f1_sum,loss_sum ,recall_sum,precision_sum = 0.0,0.0,0.0,0.0
		steps = math.ceil(len(data.train_mask)/batch_size)
		model.train()
		random.shuffle(data.train_mask)
		train_loss_sum = 0.0
		for step in range(steps):
			optimizer.zero_grad(set_to_none=True)
			if step == steps-1:
				train_edge_id = data.train_mask[step*batch_size:]
			else:
				train_edge_id = data.train_mask[step*batch_size:(step+1)*batch_size]
			
			#output = model(x=data.embed1,edge_index=data.edge2,sparse_adj=data.sparse_adj2,edge_id=train_edge_id) #edge index: list
			output = model(data=data,edge_id=train_edge_id)
			label = data.edge_attr[train_edge_id]
			label = label.type(torch.FloatTensor).to(device)
			loss = loss_fn(output,label)
			if not torch.isfinite(loss):
				raise RuntimeError(f'non-finite training loss at epoch {epoch+1}, step {step+1}')
			train_loss_sum += loss.item()
			if use_grad_conflict_correction:
				task_losses = build_labelwise_bce_losses(output, label, loss_fn)
				apply_label_agnostic_gradient_conflict(model, task_losses)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				optimizer.step()
			elif use_amp:
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				scaler.step(optimizer)
				scaler.update()
			else:
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				optimizer.step()
		#validation
		model.eval()	
		valid_pre_result_list = []
		valid_label_list = []
		valid_loss_sum = 0.0
		#torch.save()#save model
		steps = math.ceil(len(data.val_mask) / batch_size)
		saved_pred = []

		with torch.no_grad(): #validation set
			for step in range(steps):
				if step == steps-1:
					valid_edge_id = data.val_mask[step*batch_size:]
				else:
					valid_edge_id = data.val_mask[step*batch_size:(step+1)*batch_size]

				output = model(data=data,edge_id=valid_edge_id)

				label = data.edge_attr[valid_edge_id]
				label = label.type(torch.FloatTensor).to(device)
				loss = loss_fn(output,label)
				valid_loss_sum += loss.item()

				m = nn.Sigmoid()
				score = m(output)

				valid_label_list.append(label.cpu().data)
				saved_pred.append(score.to(device).cpu().data )


		valid_label_list = torch.cat(valid_label_list, dim=0)
		saved_pred = torch.cat(saved_pred,dim=0)
		y_true_np = valid_label_list.numpy().astype(int)
		y_score_np = saved_pred.numpy().astype(float)
		threshold_state = select_eval_thresholds(y_true_np, y_score_np, args)
		y_pred_np = apply_thresholds(y_score_np, threshold_state["thresholds"])
		valid_pre_result_list = torch.tensor(y_pred_np, dtype=torch.float)

		metrics = utils.Metrictor_PPI(valid_pre_result_list, valid_label_list)
		record = metrics.append_result(result_prefix+'.txt',epoch+1,train_loss_sum,valid_loss_sum)
		threshold_note = f", threshold_mode {threshold_state['mode']}, threshold {threshold_state['display']}"
		record += threshold_note
		ltda_summary = None
		ltda_detail_lines = []
		if bool(getattr(args, 'use_ltda', False)):
			ltda_stats = collect_ltda_debug_stats(model, node_degree=getattr(data, 'link_degree', None))
			ltda_summary, ltda_detail_lines = summarize_ltda_debug_stats(ltda_stats)
			if ltda_summary is not None:
				record += f", {ltda_summary}"
		with open(result_prefix+'.txt','a') as f:
			f.write(f"threshold_mode {threshold_state['mode']}, threshold {threshold_state['display']}\n")
			if ltda_summary is not None:
				f.write(f"{ltda_summary}\n")
				for line in ltda_detail_lines:
					f.write(f"{line}\n")
		print(record)

		recall_sum += metrics.recall
		precision_sum += metrics.pre
		f1_sum += metrics.microF1
		loss_sum += loss.item()
		valid_loss = valid_loss_sum / steps

		if best_f1 < metrics.microF1: #epoch == epochs -1 :
			best_f1 = metrics.microF1
			best_epoch = epoch
			best_threshold_state = threshold_state
			result =  {
				'pred':saved_pred,
				'actual':valid_label_list,
				'threshold_mode':threshold_state['mode'],
				'thresholds':threshold_state['thresholds'],
			}
			if args.aly:
				torch.save(model.state_dict(),result_prefix+'_weighs.pt' )
			#torch.save()
	
	if global_best_f1 < best_f1:
		global_best_f1 = best_f1
		torch.save(result,result_prefix+'.pt')

	if args.aly:
		#model.load_state_dict(torch.load(result_prefix+'_weighs.pt', weights_only=True))
		if hasattr(model, 'compute_hiearchical_level'):
			aly_data = model.compute_hiearchical_level(data=data)
		else:
			print('warning: compute_hiearchical_level is not implemented; skipping legacy hierarchy export')


	return global_best_f1, aly_data, best_epoch, best_threshold_state



def get_args_parser():
	parser = argparse.ArgumentParser('PwPPI',add_help=False)
	parser.add_argument('-m',default=None,type=str,help='mode, optinal value: bfs,dfs,rand,read,data')
	# parser.add_argument('-m',default='s',type=str,help='mode')
	parser.add_argument('-o',default='output',type=str)
	parser.add_argument('-t', default='LTDA', type=str,help='model type; defaults to LTDA')
	# parser.add_argument('-sf', default=None,type=str,help='optional input, contains path for sequence and relation file')
	parser.add_argument('-i',default=None,type=str,help='path for sequnce and relation file')
	parser.add_argument('-i1',default=None,type=str,help='sequence file')
	parser.add_argument('-i2',default=None,type=str,help='relation file')
	parser.add_argument('-i3',default=None,type=str,help='file path of test set indices (for read mode)')
	parser.add_argument('-i4',default=None,type=str,help='prefix for the pre generated structure')
	parser.add_argument('-s1',default='/home/user1/code/PPI4/data/structure_data',type=str,help='file path for structure file')
	#parser.add_argument('-i4',default='../data/map1.csv',type=str,help='file path for map STRING id to uniref id')
	parser.add_argument('-e',default=50,type=int,help='epochs')
	parser.add_argument('-b', default=256, type=int,help='batch size')
	parser.add_argument('-ln', default=3, type=int,help='graph layer num')
	parser.add_argument('-L', default=128, type=int,help='length for sequence padding')
	parser.add_argument('-Loss', default='CE', type=str,help='loss function')
	parser.add_argument('-ff', default='CnM', type=str,help='feature fusion option, default mul')
	parser.add_argument('-hl', default=512, type=int,help='hidden layer')
	parser.add_argument('-sv',default=False,type=str2bool,help='if save dataset path')
	parser.add_argument('-cuda',default=False,type=str2bool,help='if use cuda')
	parser.add_argument('-amp', default=False, type=str2bool, help='if use mixed precision training')
	parser.add_argument('-force',default=True,type=str2bool,help='if write to existed output file')
	parser.add_argument('-use_att', default=False, type=str2bool, help='if use attention in graph aggregation')
	parser.add_argument('-local_agg', default=False, type=str2bool, help='if use local attention aggregation')
	parser.add_argument('-lorentz_nonlin_mode', default='legacy', choices=['legacy', 'none', 'tangent'], help='Lorentz nonlinearity mode: legacy applies ReLU on Lorentz coordinates after layer 1, none disables it, tangent applies it in tangent space')
	parser.add_argument('-pre_diffusion_mode', default='none', choices=['none', 'euclidean_input'], help='pre-encoder feature diffusion mode; euclidean_input diffuses raw Euclidean node features before mapping to the manifold')
	parser.add_argument('-label_specific_heads', default=False, type=str2bool, help='if use shared MLP with label residual adapter instead of only one shared classifier head')
	parser.add_argument('-use_degree_conditioned_gate', default=False, type=str2bool, help='if use degree-conditioned interaction gating for pair fusion')
	parser.add_argument('-degree_num_buckets', default=5, type=int, help='number of degree buckets for degree-conditioned interaction gating')
	parser.add_argument('-use_grad_conflict_correction', default=False, type=str2bool, help='if apply label-agnostic gradient conflict correction across multilabel BCE losses')
	parser.add_argument('-bce_pos_weight_mode', default='none', choices=['none', 'auto', 'manual'], help='class weighting mode for BCEWithLogitsLoss')
	parser.add_argument('-bce_pos_weight_values', default=None, type=str2floatlist, help='comma separated positive-class weights for BCEWithLogitsLoss when mode=manual')
	parser.add_argument('-bce_pos_weight_clip', default=0.0, type=float, help='maximum positive-class weight for BCEWithLogitsLoss; <=0 disables clipping')
	parser.add_argument('-eval_threshold_mode', default='fixed', choices=['fixed', 'global_search', 'per_label_search'], help='threshold strategy for validation F1/ACC')
	parser.add_argument('-eval_threshold', default=0.5, type=float, help='fixed threshold for validation F1/ACC when eval_threshold_mode=fixed')
	parser.add_argument('-eval_threshold_grid', default='0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95', type=str2floatlist, help='comma separated threshold grid for global/per-label validation search')
	parser.add_argument('-use_ltda', default=False, type=str2bool, help='if use LTDA before decoder in Lorentz branches')
	parser.add_argument('-ltda_scales', default='1,2,3', type=str2intlist, help='comma separated diffusion scales for LTDA')
	parser.add_argument('-ltda_transition', default='uniform', choices=['curvature', 'uniform', 'geometric'], help='transition kernel for LTDA: curvature-aware, row-wise uniform, or geometry-aware')
	parser.add_argument('-ltda_self_loop', default=True, type=str2bool, help='if include self-loop transitions inside LTDA diffusion; disable to keep only the outer residual self-preservation')
	parser.add_argument('-ltda_return_manifold', default=True, type=str2bool, help='if map tangent-diffused features back to the manifold before decoder; disable to keep diffusion outputs in tangent space')
	parser.add_argument('-ltda_alpha_init', default=2.0, type=float, help='initial logit for LTDA residual strength')
	parser.add_argument('-ltda_deg_slope_init', default=-20, type=float, help='initial slope for degree-aware LTDA attenuation')
	parser.add_argument('-ltda_curv_clip', default=5.0, type=float, help='clip value for locally standardized Forman curvature in LTDA')
	parser.add_argument('-ltda_shuffle_curv', default=False, type=str2bool, help='if shuffle local curvature scores in LTDA for control experiments')
	parser.add_argument('-use_diffusion', default=False, type=str2bool, help='if use wavelet diffusion before GIN in Lorentz branches')
	parser.add_argument('-diff_scales', default='1,2,3', type=str2intlist, help='comma separated diffusion scales')
	parser.add_argument('-diff_alpha_init', default=0.0, type=float, help='initial logit for diffusion residual strength')
	parser.add_argument('-diff_a_init', default=-0.5, type=float, help='initial bias for degree-aware diffusion attenuation')
	parser.add_argument('-diff_b_init', default=0.5, type=float, help='initial slope for degree-aware diffusion attenuation')
	parser.add_argument('-edge_dropout', default=0.0, type=float, help='drop rate applied to sparse_adj1 and edge1 during training only')
	parser.add_argument('-edge_dropout_mode', default='uniform', choices=['uniform', 'hub_aware', 'hub_hub_only'], help='edge dropout mode for Lorentz/Hyperbolic relation graphs')
	parser.add_argument('-edge_dropout_hub_alpha', default=0.2, type=float, help='extra hub-aware drop strength added on top of base edge_dropout')
	parser.add_argument('-anchor_topk', default=5, type=int, help='top-k anchors used for zero-degree endpoint imputation')
	parser.add_argument('-anchor_tau', default=0.1, type=float, help='softmax temperature for anchor weighting')
	parser.add_argument('-anchor_max_degree', default=-1.0, type=float, help='maximum link degree allowed for anchor candidates; <=0 disables the filter')
	parser.add_argument('-partner_hub_threshold', default=30.0, type=float, help='threshold used to flag zero-high partner patterns in pair gating')
	parser.add_argument('-mainfold',default='Hyperboloid',type=str,help='any of the following: Euclidean, Hyperboloid, PoincareBall, Lorentz')
	parser.add_argument('-pr',default=0.0,type=float,help='perturbation ratio')
	parser.add_argument('-sf',default=None,type=str,help='pfolder that contain pdb file')
	parser.add_argument('-seed', default=7, type=int, help='random seed for split, shuffle, and torch randomness')

	parser.add_argument('-aly', default=False, type=str2bool, help='if analyze hierarchical embedding')
	return parser

# python3 main.py -m data -i1 [seq file] -i2 [interaction file]

# python main.py -m bfs -t LTDA -i data/27K.txt -i4 features/27K/27K -o lorentz_smoke -e 100 -b 2048 -mainfold Lorentz -use_att true -local_agg true -ln 3 -use_diffusion true -diff_scales 1,2,3
#python3 main.py -m bfs -t LTDA -i data/27K.txt -i4 features/27K -o test -e 100 -mainfold Hyperboloid
#python3 main.py -m bfs -t LTDA -i 27K.txt -i4 features/27K -o test -e 100 -mainfold Hyperboloid
def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(torch.cuda.is_available())
	print(device)
	set_seed(args.seed)
	print(f'seed: {args.seed}')

	if args.i:
		with open(args.i,'r') as f:
			args.i1 = f.readline().strip()
			args.i2 = f.readline().strip()
	
	if not args.force:
		if os.path.isfile(args.o+'.txt'):
			print('output name already exists')
			exit()

	if args.m == 'data':
		ppi_data.generate_structure_feature(args)
		return

	PPIData = ppi_data.PPIData(args)
	data = PPIData.data
	data.to(device)
	if args.t in ['LTDA', 'HI-PPI']:
		model = Models.LTDA(
			data.embed1.shape[-1],
			args=args,
			layer_num=args.ln,
			in_len=args.L,
			use_att=args.use_att,
			local_agg=args.local_agg,
		).to(device)	
	elif args.t == 'ab1':
		model = Models.ablation1(
			data.embed1.shape[-1],
			args=args,
			layer_num=args.ln,
			in_len=args.L,
			use_att=args.use_att,
			local_agg=args.local_agg,
		).to(device)
	elif args.t == 'ab2':
		model = Models.ablation2(
			data.embed1.shape[-1],
			args=args,
			layer_num=args.ln,
			in_len=args.L,
			use_att=args.use_att,
			local_agg=args.local_agg,
		).to(device)

	if args.Loss=='CE':
		pos_weight = resolve_bce_pos_weight(data, args, device)
		if pos_weight is None:
			loss_fn = nn.BCEWithLogitsLoss().to(device)
			print('bce_pos_weight: none')
		else:
			loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
			print(f'bce_pos_weight ({args.bce_pos_weight_mode}): {pos_weight.detach().cpu().tolist()}')
	elif args.Loss=='AS':
		loss_fn = Models.AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True).to(device)
	if args.use_grad_conflict_correction and args.Loss != 'CE':
		raise ValueError('use_grad_conflict_correction currently supports only BCEWithLogitsLoss (Loss=CE)')
	optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)


	start = time.time()
	best_f1,aly_data,best_epoch,best_threshold_state = train(model,data,loss_fn,optimizer,device,args.o,args.b,args.e,None,0.0,args)
	end = time.time()
	mins = (end - start)/60

	print(f'\nbest F1 score: {best_f1:.4f} (epoch {best_epoch+1}, threshold_mode {best_threshold_state["mode"]}, threshold {best_threshold_state["display"]}), running time: {mins:.2f} minutes')
	print(f'output save to {args.o}.txt')
	if args.o:
		with open(args.o+'.txt','r+') as file:
			file_data = file.read()
			file.seek(0,0)
			command = ' '.join(arg for arg in sys.argv)
			line = f'command: {command}\n'
			line += f'best F1 score: {best_f1:.4f}\n'
			line += f'training time: {mins:.2f} minutes\n'
			line += f'mode: {args.m}\n'		
			line += f'layer num: {args.ln}\n'	
			line += f'filePath: {args.i1} {args.i2}\n'
			if args.i3:
				line += f'valid set path: {args.i3}\n'
			line += f'model: {args.t}\n'
			line +=f'Loss function: {args.Loss}\n'
			line +=f'max length of seqs: {args.L}\n'
			#line +=f'feature 1 shape {graph.f1.size()}\n'# feature 2 shape {graph.f2.size()}\n'
			line +=f'epoch: {args.e}\n'
			line +=f'feature fusion mode: {args.ff}\n'
			line +=f'mainfold: {args.mainfold}\n'
			line +=f'use_att: {args.use_att}\n'
			line +=f'local_agg: {args.local_agg}\n'
			line +=f'lorentz_nonlin_mode: {args.lorentz_nonlin_mode}\n'
			line +=f'pre_diffusion_mode: {args.pre_diffusion_mode}\n'
			line +=f'label_specific_heads: {args.label_specific_heads}\n'
			line +=f'use_degree_conditioned_gate: {args.use_degree_conditioned_gate}\n'
			line +=f'degree_num_buckets: {args.degree_num_buckets}\n'
			line +=f'use_grad_conflict_correction: {args.use_grad_conflict_correction}\n'
			line +=f'bce_pos_weight_mode: {args.bce_pos_weight_mode}\n'
			line +=f'bce_pos_weight_values: {args.bce_pos_weight_values}\n'
			line +=f'bce_pos_weight_clip: {args.bce_pos_weight_clip}\n'
			line +=f'eval_threshold_mode: {args.eval_threshold_mode}\n'
			line +=f'eval_threshold: {args.eval_threshold}\n'
			line +=f'eval_threshold_grid: {args.eval_threshold_grid}\n'
			line +=f'use_ltda: {args.use_ltda}\n'
			line +=f'ltda_scales: {args.ltda_scales}\n'
			line +=f'ltda_transition: {args.ltda_transition}\n'
			line +=f'ltda_self_loop: {args.ltda_self_loop}\n'
			line +=f'ltda_return_manifold: {args.ltda_return_manifold}\n'
			line +=f'ltda_alpha_init: {args.ltda_alpha_init}\n'
			line +=f'ltda_deg_slope_init: {args.ltda_deg_slope_init}\n'
			line +=f'ltda_curv_clip: {args.ltda_curv_clip}\n'
			line +=f'ltda_shuffle_curv: {args.ltda_shuffle_curv}\n'
			line +=f'use_diffusion: {args.use_diffusion}\n'
			line +=f'diff_scales: {args.diff_scales}\n'
			line +=f'diff_alpha_init: {args.diff_alpha_init}\n'
			line +=f'diff_a_init: {args.diff_a_init}\n'
			line +=f'diff_b_init: {args.diff_b_init}\n'
			line +=f'edge_dropout: {args.edge_dropout}\n'
			line +=f'edge_dropout_mode: {args.edge_dropout_mode}\n'
			line +=f'edge_dropout_hub_alpha: {args.edge_dropout_hub_alpha}\n'
			line +=f'anchor_topk: {args.anchor_topk}\n'
			line +=f'anchor_tau: {args.anchor_tau}\n'
			line +=f'anchor_max_degree: {args.anchor_max_degree}\n'
			line +=f'partner_hub_threshold: {args.partner_hub_threshold}\n'
			line +=f'seed: {args.seed}\n'
			line +=f'amp: {args.amp}\n'
			
			file.write(line + '\n' + file_data)


	if args.aly:
		if aly_data is not None:
			PPIData.analyze_hiearchical(aly_data)
		else:
			print('warning: hierarchy tensor is unavailable; skipping analyze_hiearchical')

	return best_f1

#python3 main.py -m read -t HPPI19 -i 27K.txt -i3 /home/user1/code/PPIKG/multiSet/o2/27k_bfs10.data -i4 features/27K -o ../result/test -e 100 -mainfold Hyperboloid
if __name__ == "__main__":
	parser = argparse.ArgumentParser('PPIM', parents=[get_args_parser()])
	args = parser.parse_args()	
	best_f1 = main(args)
