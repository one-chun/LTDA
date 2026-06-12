import torch
import torch.nn as nn
import math
import random
import torch
import torch_geometric
import torch.nn.functional as F
import torch_geometric.nn.conv as Conv
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax as pyg_softmax
import numpy as np
import dgl
from torch.nn import Parameter
from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv
import mainfold


def manifold_uses_time_coordinate(manifold_name):
	return manifold_name in ['Hyperboloid', 'Lorentz']


class LTDA(nn.Module):
	def __init__(self,input_dim,args=None,act='relu',layer_num=2,radius=None,dropout=0.0,if_bias=True,use_att=0,local_agg=0,feature_fusion='CnM',class_num =7,in_len=512,device=None):
		super(LTDA, self).__init__()
		self.models = torch.nn.ModuleList()#seven independent GNN models
		self.layer_num = layer_num
		self.class_num = class_num
		self.feature_fusion = feature_fusion
		#self.f1_transform = 64
		self.layer_num = layer_num
		self.in_len = in_len
		self.input_dim = input_dim
		#self.long_conv = hyena.HyenaOperator(d_model=input_dim,l_max=in_len)#
		#self.fc1 = nn.Linear(math.floor( in_len / pool_size),self.f1_transform )
		self.hyper_dim = int(self.input_dim/2)
		self.mainfold_name = args.mainfold
		self.mainfold = getattr(mainfold,self.mainfold_name)()#mainfold.Hyperboloid()#
		self.feature_fusion = feature_fusion
		self.layer_num = layer_num
		self.device = device
		self.label_specific_heads = bool(getattr(args, 'label_specific_heads', False))
		self.use_degree_conditioned_gate = bool(getattr(args, 'use_degree_conditioned_gate', False))
		self.degree_num_buckets = int(getattr(args, 'degree_num_buckets', 5))
		self.lorentz_nonlin_mode = getattr(args, 'lorentz_nonlin_mode', 'legacy')
		self.edge_dropout = float(getattr(args, 'edge_dropout', 0.0))
		self.edge_dropout_mode = getattr(args, 'edge_dropout_mode', 'uniform')
		self.edge_dropout_hub_alpha = float(getattr(args, 'edge_dropout_hub_alpha', 0.2))
		self.pre_diffusion_mode = getattr(args, 'pre_diffusion_mode', 'none')
		if self.pre_diffusion_mode not in ['none', 'euclidean_input']:
			raise ValueError(f'Unsupported pre_diffusion_mode: {self.pre_diffusion_mode}')
		self.use_ltda = bool(getattr(args, 'use_ltda', False)) and self.mainfold.name in ['Lorentz', 'Hyperboloid']
		self.ltda_scales = getattr(args, 'ltda_scales', [1, 2, 3])
		self.ltda_transition = getattr(args, 'ltda_transition', 'curvature')
		self.ltda_return_manifold = bool(getattr(args, 'ltda_return_manifold', True))
		self.ltda_alpha_init = float(getattr(args, 'ltda_alpha_init', -2.0))
		self.ltda_deg_slope_init = float(getattr(args, 'ltda_deg_slope_init', 0.5))
		self.ltda_curv_clip = float(getattr(args, 'ltda_curv_clip', 5.0))
		self.ltda_shuffle_curv = bool(getattr(args, 'ltda_shuffle_curv', False))
		self.ltda_self_loop = bool(getattr(args, 'ltda_self_loop', True))
		self.use_diffusion = bool(getattr(args, 'use_diffusion', True)) and self.mainfold.name == 'Lorentz'
		self.diff_scales = getattr(args, 'diff_scales', [1, 2, 4])
		self.diff_alpha_init = float(getattr(args, 'diff_alpha_init', -3.0))
		self.diff_a_init = float(getattr(args, 'diff_a_init', -0.5))
		if self.diff_a_init == -0.5 and self.diff_alpha_init != 0.0:
			self.diff_a_init = self.diff_alpha_init
		self.diff_b_init = float(getattr(args, 'diff_b_init', 0.5))
		dims = [self.input_dim] + ([self.hyper_dim] * (layer_num))

		if manifold_uses_time_coordinate(self.mainfold.name):
			dims[0] += 1		
		self.node_feature_dim = dims[0]
		self.anchor_topk = int(getattr(args, 'anchor_topk', 5))
		self.anchor_tau = float(getattr(args, 'anchor_tau', 0.1))
		self.anchor_max_degree = float(getattr(args, 'anchor_max_degree', -1.0))
		if self.anchor_max_degree <= 0.0:
			self.anchor_max_degree = None
		self.partner_hub_threshold = float(getattr(args, 'partner_hub_threshold', 30.0))
		if self.pre_diffusion_mode == 'euclidean_input':
			self.PreInputDiffusion = GatedWaveletDiffusion(
				self.input_dim,
				self.ltda_scales,
				init_a=self.ltda_alpha_init,
				init_b=self.ltda_deg_slope_init,
			)
		n_curvatures = len(dims)+1
		self.radius = radius
		if radius is None:
			self.curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
		else:
			self.curvatures = [torch.tensor([radius]) for _ in range(n_curvatures)]         # fixed curvature
		#self.curvatures.append(self.radius)

		act = getattr(torch.nn.functional, act)
		acts = [act] * (layer_num)
		for c in range(class_num):
			graph_layers = []
			for i in range(layer_num):
				in_dim, out_dim = dims[i], dims[i+1]
				if self.mainfold.name == 'Lorentz':
					if self.lorentz_nonlin_mode == 'legacy':
						nonlin = acts[i] if i != 0 else None
					elif self.lorentz_nonlin_mode == 'none':
						nonlin = None
					elif self.lorentz_nonlin_mode == 'tangent':
						nonlin = acts[i]
					else:
						raise ValueError(f'Unknown lorentz_nonlin_mode: {self.lorentz_nonlin_mode}')
					graph_layers.append(
						LorentzGraphConvolution(
							self.mainfold,
							in_dim,
							out_dim,
							if_bias,
							dropout,
							use_att,
							local_agg,
							nonlin,
							nonlin_mode=self.lorentz_nonlin_mode,
						)
					)
				else:
					c_in, c_out = self.curvatures[i+1], self.curvatures[i+2]
					graph_layers.append(HyperbolicGCN(self.mainfold,in_dim,out_dim,c_in, c_out,dropout,acts[i],if_bias,use_att,local_agg))
			if self.use_ltda:
				ltda_c = None
				if self.mainfold.name == 'Hyperboloid':
					ltda_c = self.curvatures[-1]
				graph_layers.append(
					CurvatureHyperbolicDiffusionKernel(
						self.mainfold,
						dims[-1],
						manifold_c=ltda_c,
						transition_mode=self.ltda_transition,
						scales=self.ltda_scales,
						alpha_init=self.ltda_alpha_init,
						deg_slope_init=self.ltda_deg_slope_init,
						curv_clip=self.ltda_curv_clip,
						shuffle_curv=self.ltda_shuffle_curv,
						use_self_loop=self.ltda_self_loop,
						return_manifold=self.ltda_return_manifold,
					)
				)
			if self.use_ltda and not self.ltda_return_manifold:
				graph_layers.append(TangentSpaceDecoder(dims[-1] - 1, dims[-1], if_bias, dropout))
			else:
				graph_layers.append(HyperbolicDecoder(self.mainfold_name,dims[-2],dims[-1],if_bias,dropout,self.curvatures[-1]))
			if self.use_diffusion:
				graph_layers.append(GatedWaveletDiffusion(dims[-1], self.diff_scales, init_a=self.diff_a_init, init_b=self.diff_b_init))
			graph_layers.append(torch_geometric.nn.models.GIN(dims[-1],dims[-1],1,out_dim,act='tanh',norm=nn.BatchNorm1d(dims[-1])))			
			self.models.append(nn.Sequential(*graph_layers))

		hidden3 = dims[0]+1*class_num*dims[-1]	
		self.degree_feature_dim = 1 + self.degree_num_buckets + 2
		self.GatedNetwork = GatedInteractionNetwork(
			hidden3,
			hidden3,
			hidden3,
			use_degree_conditioned_gate=self.use_degree_conditioned_gate,
			degree_feature_dim=self.degree_feature_dim,
		)
		self.anchor_alpha = nn.Parameter(torch.tensor(-2.0))
		self.PairDegreeGate = nn.Sequential(
			nn.Linear(hidden3 + 6, hidden3),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden3, hidden3),
			nn.Sigmoid(),
		)
		#self.fc2 = get_classifier(hidden3,class_num,feature_fusion)
		fc2_dim = hidden3*1
		if self.label_specific_heads:
			self.shared_fc = self._build_classifier_head(fc2_dim, class_num)
			self.label_adapter = nn.Sequential(
				nn.Linear(fc2_dim, int(fc2_dim / 4)),
				nn.ReLU(),
				nn.Linear(int(fc2_dim / 4), class_num),
			)
			self.adapter_alpha = nn.Parameter(torch.tensor(-3.0))
		else:
			self.fc2 = self._build_classifier_head(fc2_dim, class_num)
		return

	def _build_classifier_head(self, input_dim, output_dim):
		hidden_dim1 = int(input_dim / 2)
		hidden_dim2 = int(input_dim / 4)
		return nn.Sequential(
		  nn.Linear(input_dim, hidden_dim1),
		  nn.ReLU(),
		  nn.Linear(hidden_dim1, hidden_dim2),
		  nn.ReLU(),
		  nn.Linear(hidden_dim2, output_dim),
		)

	def _to_hyperbolic_input(self, f1):
		if manifold_uses_time_coordinate(self.mainfold_name):
			o = torch.zeros_like(f1)
			f1 = torch.cat([o[:, 0:1], f1], dim=1)
		x_tan = self.mainfold.proj_tan0(f1, self.curvatures[0])
		x_hyp = self.mainfold.expmap0(x_tan, c=self.curvatures[0])
		x_hyp = self.mainfold.proj(x_hyp, c=self.curvatures[0])
		return f1, x_hyp

	def _edge_index_to_sparse(self, edge_index, node_num, device, dtype):
		if edge_index.numel() == 0:
			empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
			empty_val = torch.empty((0,), dtype=dtype, device=device)
			return torch.sparse_coo_tensor(
				empty_idx,
				empty_val,
				(node_num, node_num),
				device=device,
				dtype=dtype,
			).coalesce()
		values = torch.ones(edge_index.size(1), device=device, dtype=dtype)
		return torch.sparse_coo_tensor(
			edge_index.to(device=device),
			values,
			(node_num, node_num),
			device=device,
			dtype=dtype,
		).coalesce()

	def _edge_drop_prob(self, edge_index, node_degree, drop_rate):
		edge_num = edge_index.size(1)
		has_hub_component = self.edge_dropout_mode in ['hub_aware', 'hub_hub_only'] and self.edge_dropout_hub_alpha > 0.0
		if drop_rate <= 0.0 and not has_hub_component:
			return torch.zeros(edge_num, device=edge_index.device, dtype=torch.float)
		base = torch.full((edge_num,), float(drop_rate), device=edge_index.device, dtype=torch.float)
		if node_degree is None or self.edge_dropout_mode == 'uniform':
			return base.clamp(max=0.95)

		src, dst = edge_index[0], edge_index[1]
		log_deg = torch.log1p(node_degree.float())
		max_log_deg = log_deg.max().clamp_min(1.0)
		norm_src = (log_deg[src] / max_log_deg).to(base.device)
		norm_dst = (log_deg[dst] / max_log_deg).to(base.device)

		if self.edge_dropout_mode == 'hub_aware':
			hub_score = norm_src * norm_dst
			return (base + self.edge_dropout_hub_alpha * hub_score).clamp(max=0.8)

		if self.edge_dropout_mode == 'hub_hub_only':
			hub_mask = (norm_src >= 0.8) & (norm_dst >= 0.8)
			drop_prob = base.clone()
			drop_prob[hub_mask] = (drop_prob[hub_mask] + self.edge_dropout_hub_alpha).clamp(max=0.8)
			return drop_prob

		return base.clamp(max=0.95)

	def _drop_edge_data(self, adj, edge_index, drop_rate, node_degree=None):
		has_hub_component = self.edge_dropout_mode in ['hub_aware', 'hub_hub_only'] and self.edge_dropout_hub_alpha > 0.0
		if drop_rate <= 0.0 and not has_hub_component:
			return adj, edge_index

		drop_prob = self._edge_drop_prob(edge_index, node_degree, drop_rate)
		keep = torch.rand_like(drop_prob) > drop_prob
		if keep.sum() == 0:
			return adj, edge_index

		new_edge_index = edge_index[:, keep]
		new_adj = self._edge_index_to_sparse(
			new_edge_index,
			adj.size(0),
			adj.device,
			adj.dtype,
		)
		return new_adj, new_edge_index

	def _maybe_edge_dropout(self, sparse_adj, edges, node_degree=None):
		has_hub_component = self.edge_dropout_mode in ['hub_aware', 'hub_hub_only'] and self.edge_dropout_hub_alpha > 0.0
		if (not self.training) or (self.edge_dropout <= 0.0 and not has_hub_component):
			return sparse_adj, edges

		dropped_sparse_adj = []
		dropped_edges = []
		for adj, edge in zip(sparse_adj, edges):
			new_adj, new_edge = self._drop_edge_data(adj, edge, self.edge_dropout, node_degree=node_degree)
			dropped_sparse_adj.append(new_adj)
			dropped_edges.append(new_edge)
		return dropped_sparse_adj, dropped_edges

	def _degree_threshold_from_quantile(self, values, quantile):
		if values.numel() == 0:
			return values.new_tensor(0.0)
		sorted_values, _ = torch.sort(values)
		index = int(max(0, min(sorted_values.numel() - 1, math.ceil(quantile * sorted_values.numel()) - 1)))
		return sorted_values[index]

	def _build_node_degree_features(self, node_degree):
		deg = node_degree.float()
		log_deg = torch.log1p(deg)
		max_log_deg = log_deg.max().clamp_min(1.0)
		norm_log_deg = log_deg / max_log_deg

		if self.degree_num_buckets <= 1:
			bucket_onehot = norm_log_deg.new_ones((norm_log_deg.size(0), 1))
		else:
			boundaries = torch.linspace(
				0.0,
				1.0,
				steps=self.degree_num_buckets + 1,
				device=deg.device,
				dtype=norm_log_deg.dtype,
			)[1:-1]
			bucket = torch.bucketize(norm_log_deg, boundaries)
			bucket_onehot = F.one_hot(bucket, num_classes=self.degree_num_buckets).float()

		nonzero = norm_log_deg[deg > 0]
		if nonzero.numel() == 0:
			hub_thresh = norm_log_deg.new_tensor(1.0)
			super_thresh = norm_log_deg.new_tensor(1.0)
		else:
			hub_thresh = self._degree_threshold_from_quantile(nonzero, 0.95)
			super_thresh = self._degree_threshold_from_quantile(nonzero, 0.99)
		is_hub = (norm_log_deg >= hub_thresh).float().unsqueeze(-1)
		is_super_hub = (norm_log_deg >= super_thresh).float().unsqueeze(-1)
		return torch.cat([norm_log_deg.unsqueeze(-1), bucket_onehot, is_hub, is_super_hub], dim=-1)

	def _encode_relation_branches(self, x_hyp, sparse_adj):
		branch_states = []
		for i,m in enumerate(self.models):
			tmp = x_hyp
			for j in range(self.layer_num):
				tmp, _ = m[j]((tmp,sparse_adj[i]))
			branch_states.append(tmp)
		return branch_states

	def _decode_relation_branches(self, branch_states, sparse_adj, edges, node_degree=None):
		output = []
		for i,m in enumerate(self.models):
			next_layer = self.layer_num
			branch_state = branch_states[i]
			if self.use_ltda:
				branch_state = m[next_layer](branch_state, sparse_adj[i], node_degree=node_degree)
				next_layer += 1
			tmp = m[next_layer].forward(branch_state)
			next_layer += 1
			if self.use_diffusion:
				tmp = m[next_layer](tmp, sparse_adj[i], node_degree=node_degree)
				next_layer += 1
			tmp = m[next_layer](tmp,edges[i])
			output.append(tmp)
		return output

	def _anchor_impute_for_nodes(self, x_graph, raw_feat, target_ids, train_degree):
		if target_ids.numel() == 0:
			return {}

		candidate_ids = torch.where(train_degree > 0)[0]
		if self.anchor_max_degree is not None:
			candidate_ids = candidate_ids[train_degree[candidate_ids] <= self.anchor_max_degree]
		if candidate_ids.numel() == 0:
			return {}

		raw_norm = F.normalize(raw_feat, p=2, dim=-1, eps=1e-12)
		target_norm = raw_norm[target_ids]
		candidate_norm = raw_norm[candidate_ids]
		sim = torch.matmul(target_norm, candidate_norm.T)

		k = min(self.anchor_topk, candidate_ids.numel())
		if k <= 0:
			return {}
		topv, topi = torch.topk(sim, k=k, dim=1)
		anchor_ids = candidate_ids[topi]
		temperature = max(self.anchor_tau, 1e-6)
		weight = F.softmax(topv / temperature, dim=1)

		anchor_repr = x_graph[anchor_ids]
		imputed = torch.sum(weight.unsqueeze(-1) * anchor_repr, dim=1)
		alpha = torch.sigmoid(self.anchor_alpha)
		old = x_graph[target_ids].clone()
		new = old.clone()
		new[:, self.node_feature_dim:] = old[:, self.node_feature_dim:] + alpha * (
			imputed[:, self.node_feature_dim:] - old[:, self.node_feature_dim:]
		)
		return {int(target_ids[i].item()): new[i] for i in range(target_ids.size(0))}

	def _build_degree_pair_feat(self, deg_u, deg_v):
		log_du = torch.log1p(deg_u.float())
		log_dv = torch.log1p(deg_v.float())
		min_deg = torch.minimum(log_du, log_dv)
		max_deg = torch.maximum(log_du, log_dv)
		min_raw = torch.minimum(deg_u.float(), deg_v.float())
		max_raw = torch.maximum(deg_u.float(), deg_v.float())
		one_sided_cold = ((deg_u == 0) | (deg_v == 0)).float()
		zero_high = ((min_raw == 0) & (max_raw >= self.partner_hub_threshold)).float()
		return torch.stack([
			min_deg,
			max_deg,
			torch.abs(log_du - log_dv),
			log_du + log_dv,
			one_sided_cold,
			zero_high,
		], dim=-1)

	def _build_node_representation(self, data):
		f1 = data.embed1
		sparse_adj = data.sparse_adj1
		edges = data.edge1
		node_degree = getattr(data, 'link_degree', None)
		if node_degree is not None:
			node_degree = node_degree.to(f1.device)
		if self.pre_diffusion_mode == 'euclidean_input':
			f1 = self.PreInputDiffusion(f1, data.sparse_adj2, node_degree=node_degree)
		f1, x_hyp = self._to_hyperbolic_input(f1)
		sparse_adj, edges = self._maybe_edge_dropout(sparse_adj, edges, node_degree=node_degree)
		branch_states = self._encode_relation_branches(x_hyp, sparse_adj)
		decoded = self._decode_relation_branches(branch_states, sparse_adj, edges, node_degree=node_degree)
		return torch.cat([f1] + decoded, dim=1)

	def _build_edge_embedding(self, data, edge_id):
		edge_index = data.edge2
		x = self._build_node_representation(data)
		node_id = edge_index[:, edge_id]
		u = node_id[0]
		v = node_id[1]
		x1 = x[u].clone()
		x2 = x[v].clone()

		train_degree = getattr(data, 'link_degree', None)
		if train_degree is not None:
			train_degree = train_degree.to(x.device)
			zero_nodes = torch.cat([
				u[train_degree[u] == 0],
				v[train_degree[v] == 0],
			]).unique()
			replace_dict = self._anchor_impute_for_nodes(
				x_graph=x,
				raw_feat=data.embed1,
				target_ids=zero_nodes,
				train_degree=train_degree
			)
			if len(replace_dict) > 0:
				for i in range(u.numel()):
					uid = int(u[i].item())
					vid = int(v[i].item())
					if uid in replace_dict:
						x1[i] = replace_dict[uid]
					if vid in replace_dict:
						x2[i] = replace_dict[vid]

		degree_feat_u = None
		degree_feat_v = None
		if train_degree is not None and self.use_degree_conditioned_gate:
			node_degree_feat = self._build_node_degree_features(train_degree).to(x.device)
			degree_feat_u = node_degree_feat[u]
			degree_feat_v = node_degree_feat[v]

		pair_repr = self.GatedNetwork(x1, x2, degree_feat_u=degree_feat_u, degree_feat_v=degree_feat_v)
		if train_degree is not None:
			deg_feat = self._build_degree_pair_feat(train_degree[u], train_degree[v]).to(pair_repr.device)
			deg_gate = self.PairDegreeGate(torch.cat([pair_repr, deg_feat], dim=-1))
			pair_repr = pair_repr * (0.5 + deg_gate)
		return pair_repr

	def forward(self,data,edge_id=None):
		x = self._build_edge_embedding(data, edge_id)
		if self.label_specific_heads:
			shared_logits = self.shared_fc(x)
			delta_logits = self.label_adapter(x)
			alpha = torch.sigmoid(self.adapter_alpha)
			logits = shared_logits + alpha * delta_logits
		else:
			logits = self.fc2(x)
		return logits


class HyperbolicDecoder(nn.Module):
	"""
	Decoder abstract class for node classification tasks.
	"""

	def __init__(self, mainfold_name,input_dim,output_dim,if_bias,dropout,radius):
		super(HyperbolicDecoder, self).__init__()
		self.mainfold = getattr(mainfold, mainfold_name)()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.if_bias = if_bias
		self.linear = nn.Linear(self.input_dim, self.output_dim,bias=self.if_bias)
#, dropout, lambda x: x, 
		self.radius = radius

	def forward(self, x):

		x = self.mainfold.proj_tan0(self.mainfold.logmap0(x, c=self.radius), c=self.radius)
		x = self.linear(x)
		return x


class TangentSpaceDecoder(nn.Module):
	def __init__(self, input_dim, output_dim, if_bias, dropout):
		super(TangentSpaceDecoder, self).__init__()
		self.dropout = nn.Dropout(dropout)
		self.linear = nn.Linear(input_dim, output_dim, bias=if_bias)

	def forward(self, x):
		return self.linear(self.dropout(x))


class CurvatureHyperbolicDiffusionKernel(nn.Module):
	def __init__(
		self,
		manifold,
		feature_dim,
		manifold_c=None,
		transition_mode='curvature',
		scales=(1, 2, 3),
		alpha_init=-2.0,
		deg_slope_init=0.5,
		curv_clip=5.0,
		shuffle_curv=False,
		use_self_loop=True,
		return_manifold=True,
	):
		super(CurvatureHyperbolicDiffusionKernel, self).__init__()
		self.manifold = manifold
		self.feature_dim = int(feature_dim)
		self.spatial_dim = int(feature_dim) - 1
		self.manifold_c = manifold_c
		self.transition_mode = str(transition_mode)
		self.scales = sorted({int(scale) for scale in scales if int(scale) > 0})
		self.use_self_loop = bool(use_self_loop)
		self.return_manifold = bool(return_manifold)
		if len(self.scales) == 0:
			raise ValueError('CurvatureHyperbolicDiffusionKernel requires at least one positive diffusion scale')
		if self.spatial_dim <= 0:
			raise ValueError('CurvatureHyperbolicDiffusionKernel expects hyperbolic features with time coordinate')
		if self.transition_mode not in ['curvature', 'uniform', 'geometric']:
			raise ValueError(f'Unsupported LTDA transition_mode: {self.transition_mode}')
		self.curv_clip = float(curv_clip)
		self.shuffle_curv = bool(shuffle_curv)

		self.scale_weights = nn.Parameter(torch.zeros(len(self.scales)))
		self.theta_pos = nn.Parameter(torch.tensor(0.5))
		self.theta_neg = nn.Parameter(torch.tensor(1.0))
		self.geo_dist_scale_raw = nn.Parameter(torch.tensor(0.0))
		self.geo_dir_scale = nn.Parameter(torch.tensor(0.5))
		self.geo_logit_clip = 10.0
		self.self_loop_logit = nn.Parameter(torch.tensor(0.0))
		self.alpha_logit = nn.Parameter(torch.tensor(alpha_init))
		self.deg_slope_raw = nn.Parameter(torch.tensor(deg_slope_init))
		self.input_norm = nn.LayerNorm(self.spatial_dim)
		self.output_norm = nn.LayerNorm(self.spatial_dim)

	def _scatter_mean(self, values, index, node_num):
		out = values.new_zeros(node_num)
		cnt = values.new_zeros(node_num)
		out.scatter_add_(0, index, values)
		cnt.scatter_add_(0, index, torch.ones_like(values))
		return out / cnt.clamp_min(1.0)

	def _as_sparse_tensor(self, adj, device, dtype):
		if isinstance(adj, torch.Tensor):
			if adj.is_sparse:
				return adj.coalesce().to(device=device, dtype=dtype)
			return adj.to(device=device, dtype=dtype).to_sparse().coalesce()
		raise TypeError(f'Unsupported adjacency type: {type(adj).__name__}')

	def _symmetrize_non_loop_edges(self, adj):
		row_all, col_all = adj.indices()
		non_loop = row_all != col_all
		row = row_all[non_loop]
		col = col_all[non_loop]
		if row.numel() == 0:
			return row, col
		sym_row = torch.cat([row, col], dim=0)
		sym_col = torch.cat([col, row], dim=0)
		sym_val = torch.ones(sym_row.size(0), device=sym_row.device, dtype=adj.dtype)
		sym_adj = torch.sparse_coo_tensor(
			torch.stack([sym_row, sym_col], dim=0),
			sym_val,
			adj.size(),
			device=sym_row.device,
			dtype=adj.dtype,
		).coalesce()
		return sym_adj.indices()[0], sym_adj.indices()[1]

	def _local_forman_curvature(self, row, col, node_num):
		device = row.device
		dtype = torch.float32
		ones = torch.ones_like(row, dtype=dtype, device=device)
		deg = torch.zeros(node_num, dtype=dtype, device=device)
		deg.scatter_add_(0, row, ones)
		raw_curv = 4.0 - deg[row] - deg[col]

		incident_index = torch.cat([row, col], dim=0)
		incident_curv = torch.cat([raw_curv, raw_curv], dim=0)
		mean = self._scatter_mean(incident_curv, incident_index, node_num)
		diff2 = (incident_curv - mean[incident_index]).pow(2)
		var = self._scatter_mean(diff2, incident_index, node_num)
		std = torch.sqrt(var + 1e-6)

		curv_local = 0.5 * (
			(raw_curv - mean[row]) / std[row].clamp_min(1e-6)
			+
			(raw_curv - mean[col]) / std[col].clamp_min(1e-6)
		)
		curv_local = curv_local.clamp(-self.curv_clip, self.curv_clip)
		return curv_local, deg

	def _pairwise_lorentz_distance(self, x_hyp, row, col):
		x = x_hyp[row]
		y = x_hyp[col]

		if hasattr(self.manifold, 'distance'):
			try:
				dist = self.manifold.distance(x, y, c=self.manifold_c)
			except TypeError:
				dist = self.manifold.distance(x, y)
			return dist.clamp(max=50.0)

		if hasattr(self.manifold, 'dist'):
			try:
				dist = self.manifold.dist(x, y, c=self.manifold_c)
			except TypeError:
				dist = self.manifold.dist(x, y)
			return dist.clamp(max=50.0)

		if hasattr(self.manifold, 'sqdist'):
			dist2 = self.manifold.sqdist(x, y, c=self.manifold_c)
			return dist2.clamp(min=0.0, max=2500.0).sqrt().clamp(max=50.0)

		if self.manifold_c is None:
			k = torch.tensor(1.0, dtype=x.dtype, device=x.device)
		elif torch.is_tensor(self.manifold_c):
			k = (1.0 / self.manifold_c).to(dtype=x.dtype, device=x.device)
		else:
			k = torch.tensor(1.0 / self.manifold_c, dtype=x.dtype, device=x.device)
		minkowski_dot = -x[:, 0] * y[:, 0] + (x[:, 1:] * y[:, 1:]).sum(dim=-1)
		theta = (-minkowski_dot / k).clamp_min(1.0 + 1e-6)
		dist = torch.sqrt(k).reshape(-1)[0] * torch.acosh(theta)
		return dist.clamp(max=50.0)

	def _build_geometric_transition(self, adj, x_hyp):
		adj = self._as_sparse_tensor(adj, x_hyp.device, x_hyp.dtype)
		node_num = adj.size(0)
		row, col = self._symmetrize_non_loop_edges(adj)

		if row.numel() == 0:
			if self.use_self_loop:
				loop = torch.arange(node_num, device=x_hyp.device)
				prob = torch.ones(node_num, device=x_hyp.device, dtype=x_hyp.dtype)
				indices = torch.stack([loop, loop], dim=0)
			else:
				indices = torch.empty((2, 0), device=x_hyp.device, dtype=torch.long)
				prob = torch.empty((0,), device=x_hyp.device, dtype=x_hyp.dtype)
			transition = torch.sparse_coo_tensor(
				indices,
				prob,
				(node_num, node_num),
				device=x_hyp.device,
				dtype=x_hyp.dtype,
			).coalesce()
			deg = torch.zeros(node_num, device=x_hyp.device, dtype=x_hyp.dtype)
			return transition, deg

		deg = torch.zeros(node_num, device=x_hyp.device, dtype=x_hyp.dtype)
		deg.scatter_add_(0, row, torch.ones_like(row, dtype=x_hyp.dtype))

		dist = self._pairwise_lorentz_distance(x_hyp, row, col).to(dtype=x_hyp.dtype)
		dist_scale = F.softplus(self.geo_dist_scale_raw)

		z_full = self.manifold.logmap0(x_hyp, c=self.manifold_c)
		z_space = z_full[:, 1:]
		z_norm = F.normalize(z_space, p=2, dim=-1, eps=1e-12)
		dir_sim = (z_norm[row] * z_norm[col]).sum(dim=-1)

		logits = -dist_scale * dist + self.geo_dir_scale * dir_sim
		logits = logits.clamp(-self.geo_logit_clip, self.geo_logit_clip)

		if self.use_self_loop:
			loop = torch.arange(node_num, device=x_hyp.device)
			loop_logits = self.self_loop_logit.expand(node_num).to(dtype=x_hyp.dtype)
			final_row = torch.cat([row, loop], dim=0)
			final_col = torch.cat([col, loop], dim=0)
			final_logits = torch.cat([logits, loop_logits], dim=0)
		else:
			final_row = row
			final_col = col
			final_logits = logits
		prob = pyg_softmax(final_logits, final_row, num_nodes=node_num)

		transition = torch.sparse_coo_tensor(
			torch.stack([final_row, final_col], dim=0),
			prob,
			(node_num, node_num),
			device=x_hyp.device,
			dtype=x_hyp.dtype,
		).coalesce()
		return transition, deg

	def _build_transition(self, adj, z_space):
		adj = self._as_sparse_tensor(adj, z_space.device, z_space.dtype)
		node_num = adj.size(0)
		row, col = self._symmetrize_non_loop_edges(adj)
		if row.numel() == 0:
			if self.use_self_loop:
				loop = torch.arange(node_num, device=z_space.device)
				prob = torch.ones(node_num, device=z_space.device, dtype=z_space.dtype)
				indices = torch.stack([loop, loop], dim=0)
			else:
				indices = torch.empty((2, 0), device=z_space.device, dtype=torch.long)
				prob = torch.empty((0,), device=z_space.device, dtype=z_space.dtype)
			transition = torch.sparse_coo_tensor(
				indices,
				prob,
				(node_num, node_num),
				device=z_space.device,
				dtype=z_space.dtype,
			).coalesce()
			deg = torch.zeros(node_num, device=z_space.device, dtype=z_space.dtype)
			return transition, deg

		if self.use_self_loop:
			loop = torch.arange(node_num, device=z_space.device)
			final_row = torch.cat([row, loop], dim=0)
			final_col = torch.cat([col, loop], dim=0)
		else:
			final_row = row
			final_col = col

		if self.transition_mode == 'uniform':
			ones = torch.ones(final_row.size(0), device=z_space.device, dtype=z_space.dtype)
			row_deg = torch.zeros(node_num, device=z_space.device, dtype=z_space.dtype)
			row_deg.scatter_add_(0, final_row, ones)
			prob = ones / row_deg[final_row].clamp_min(1.0)
			deg = torch.zeros(node_num, device=z_space.device, dtype=z_space.dtype)
			deg.scatter_add_(0, row, torch.ones_like(row, dtype=z_space.dtype))
		else:
			curv_local, deg = self._local_forman_curvature(row, col, node_num)
			curv_local = curv_local.to(dtype=z_space.dtype)
			deg = deg.to(dtype=z_space.dtype)

			if self.shuffle_curv:
				perm = torch.randperm(curv_local.numel(), device=curv_local.device)
				curv_local = curv_local[perm]

			logits = self.theta_pos * F.relu(curv_local) + self.theta_neg * F.relu(-curv_local)
			if self.use_self_loop:
				loop_logits = self.self_loop_logit.expand(node_num).to(dtype=z_space.dtype)
				final_logits = torch.cat([logits, loop_logits], dim=0)
			else:
				final_logits = logits
			prob = pyg_softmax(final_logits, final_row, num_nodes=node_num)

		transition = torch.sparse_coo_tensor(
			torch.stack([final_row, final_col], dim=0),
			prob,
			(node_num, node_num),
			device=z_space.device,
			dtype=z_space.dtype,
		).coalesce()
		return transition, deg

	def forward(self, x_hyp, adj, node_degree=None):
		z_full = self.manifold.logmap0(x_hyp, c=self.manifold_c)
		z_space = self.input_norm(z_full[:, 1:])
		if self.transition_mode == 'geometric':
			transition, cur_deg = self._build_geometric_transition(adj, x_hyp)
		else:
			transition, cur_deg = self._build_transition(adj, z_space)

		max_scale = self.scales[-1]
		scale_set = set(self.scales)
		diffused_list = []
		propagated = z_space
		for scale in range(1, max_scale + 1):
			propagated = torch.sparse.mm(transition, propagated)
			if scale in scale_set:
				diffused_list.append(propagated)

		weights = F.softmax(self.scale_weights, dim=0)
		fused = torch.zeros_like(z_space)
		for weight, feature in zip(weights, diffused_list):
			fused = fused + weight * feature

		if node_degree is None:
			deg = cur_deg
		else:
			deg = node_degree.to(z_space.device).to(z_space.dtype)
		deg = torch.log1p(deg).unsqueeze(-1)
		deg_slope = F.softplus(self.deg_slope_raw)
		alpha = torch.sigmoid(self.alpha_logit - deg_slope * deg)

		z_new_space = self.output_norm(z_space + alpha * fused)
		if not self.return_manifold:
			return z_new_space
		zero_time = torch.zeros(
			z_new_space.size(0),
			1,
			device=z_new_space.device,
			dtype=z_new_space.dtype,
		)
		z_new_full = torch.cat([zero_time, z_new_space], dim=-1)
		z_new_full = self.manifold.proj_tan0(z_new_full, c=self.manifold_c)
		x_new = self.manifold.expmap0(z_new_full, c=self.manifold_c)
		if hasattr(self.manifold, 'proj'):
			x_new = self.manifold.proj(x_new, c=self.manifold_c)
		return x_new


class GatedWaveletDiffusion(nn.Module):
	def __init__(self, feature_dim, scales, init_a=-0.5, init_b=0.5):
		super(GatedWaveletDiffusion, self).__init__()
		self.feature_dim = feature_dim
		self.scales = sorted({int(scale) for scale in scales if int(scale) > 0})
		if len(self.scales) == 0:
			raise ValueError('GatedWaveletDiffusion requires at least one positive diffusion scale')
		self.scale_weights = nn.Parameter(torch.zeros(len(self.scales)))
		self.diff_a = nn.Parameter(torch.tensor(init_a))
		self.diff_b_raw = nn.Parameter(torch.tensor(init_b))
		self.gru = nn.GRUCell(self.feature_dim, self.feature_dim)
		self.layer_norm = nn.LayerNorm(self.feature_dim)
		self.output_norm = nn.LayerNorm(self.feature_dim)

	def _as_sparse_tensor(self, adj, device, dtype):
		if isinstance(adj, torch.Tensor):
			if adj.is_sparse:
				return adj.coalesce().to(device=device, dtype=dtype)
			return adj.to(device=device, dtype=dtype).to_sparse().coalesce()
		raise TypeError(f'Unsupported adjacency type: {type(adj).__name__}')

	def _normalized_random_walk(self, adj, device, dtype):
		adj = self._as_sparse_tensor(adj, device, dtype)
		node_num = adj.size(0)
		self_loop = torch.arange(node_num, device=device)
		self_loop_index = torch.stack([self_loop, self_loop], dim=0)
		indices = torch.cat([adj.indices(), self_loop_index], dim=1)
		values = torch.cat([adj.values(), torch.ones(node_num, device=device, dtype=dtype)], dim=0)
		adj = torch.sparse_coo_tensor(indices, values, adj.size(), device=device, dtype=dtype).coalesce()
		row, col = adj.indices()
		degree = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1e-8)
		norm_values = adj.values() * degree[row].pow(-0.5) * degree[col].pow(-0.5)
		return torch.sparse_coo_tensor(adj.indices(), norm_values, adj.size(), device=device, dtype=dtype).coalesce()

	def forward(self, features, adj, node_degree=None):
		norm_adj = self._normalized_random_walk(adj, features.device, features.dtype)
		max_scale = self.scales[-1]
		scale_set = set(self.scales)
		wavelet_features = []
		propagated = features
		for scale in range(1, max_scale + 1):
			propagated = torch.sparse.mm(norm_adj, propagated)
			if scale in scale_set:
				wavelet_features.append(propagated)

		scale_weights = F.softmax(self.scale_weights, dim=0)
		fused_feature = torch.zeros_like(features)
		hidden_state = None
		for weight, feature in zip(scale_weights, wavelet_features):
			norm_feature = self.layer_norm(feature)
			if hidden_state is None:
				hidden_state = self.gru(norm_feature, torch.zeros_like(norm_feature))
			else:
				hidden_state = self.gru(norm_feature, hidden_state)
			fused_feature = fused_feature + weight * hidden_state

		if node_degree is None:
			alpha = torch.sigmoid(self.diff_a)
		else:
			deg = torch.log1p(node_degree).to(features.device).to(features.dtype).unsqueeze(-1)
			b = F.softplus(self.diff_b_raw)
			alpha = torch.sigmoid(self.diff_a - b * deg)
		return self.output_norm(features + alpha * fused_feature)


# Backward-compatible alias for existing scripts and checkpoints.
HIPPI = LTDA


#GIN for ablation study
class ablation1(nn.Module):
	def __init__(self,input_dim,args=None,act='relu',layer_num=2,radius=None,dropout=0.0,bias=1,use_att=0,local_agg=0,feature_fusion='CnM',class_num =7,in_len=512):
		super(ablation1, self).__init__()
		self.models = torch.nn.ModuleList()#seven independent GNN models
		self.layer_num = layer_num
		self.class_num = class_num
		self.feature_fusion = feature_fusion
		#self.f1_transform = 64
		self.layer_num = layer_num
		self.in_len = in_len
		self.input_dim = input_dim
		#self.long_conv = hyena.HyenaOperator(d_model=input_dim,l_max=in_len)#
		#self.fc1 = nn.Linear(math.floor( in_len / pool_size),self.f1_transform )
		self.hyper_dim = int(self.input_dim)
		self.mainfold_name = args.mainfold
		self.mainfold = getattr(mainfold,self.mainfold_name)()
		self.feature_fusion = feature_fusion
		self.layer_num = layer_num

		dims = [self.input_dim] + ([self.hyper_dim] * (layer_num))

		if self.mainfold.name == 'Hyperboloid':
			dims[0] += 1		
		n_curvatures = len(dims)
		self.radius = radius
		if radius is None:
			self.curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
		else:
			self.curvatures = [torch.tensor([radius]) for _ in range(n_curvatures)]         # fixed curvature
		self.curvatures.append(self.radius)

		act = getattr(torch.nn.functional, act)
		acts = [act] * (layer_num)
		for c in range(class_num):
			graph_layers = []
			for i in range(layer_num):
				in_dim, out_dim = dims[i], dims[i+1]
				graph_layers.append(torch_geometric.nn.models.GIN(in_dim,out_dim,1,out_dim,act='tanh',norm=nn.BatchNorm1d(out_dim)))
			self.models.append(nn.Sequential(*graph_layers))

		hidden3 = dims[0]+class_num*sum(dims[1:])	
		self.merge = GatedInteractionNetwork(hidden3,hidden3,hidden3)
		#self.fc2 = get_classifier(hidden3,class_num,feature_fusion)
		fc2_dim = hidden3*1
		self.fc2 = nn.Sequential(
		  nn.Linear(fc2_dim,int(fc2_dim/2)),
		  nn.ReLU(),
		  nn.Linear(int(fc2_dim/2),int(fc2_dim/4)),
		  nn.ReLU(),
		  nn.Linear(int(fc2_dim/4),class_num),
		)
		return

	def forward(self,data,edge_id=None):
		f1 = data.embed1 #f1 = data.encode1
		sparse_adj = data.sparse_adj1
		edges = data.edge1
		edge_index = data.edge2
		#f1 = self.fc1(f1)

		if self.mainfold_name == 'Hyperboloid':
			o = torch.zeros_like(f1)
			f1 = torch.cat([o[:, 0:1], f1], dim=1)
		output = [f1]

		for i,m in enumerate(self.models):
			tmp = f1
			for j in range(self.layer_num):
				tmp =  m[-1](tmp,edges[i])
				output.append(tmp)


		x = torch.cat(output,dim=1)
		node_id = edge_index[:, edge_id]
		x1 = x[node_id[0]]
		x2 = x[node_id[1]]

		x = torch.cat([self.merge(x1, x2)], dim=1) #torch.mul(x1, x2)
		x = self.fc2(x)
		return x

#abltion2
class ablation2(nn.Module):
	def __init__(self,input_dim,args=None,act='relu',layer_num=2,radius=None,dropout=0.0,bias=1,use_att=0,local_agg=0,feature_fusion='CnM',class_num =7,in_len=512):
		super(ablation2, self).__init__()
		self.models = torch.nn.ModuleList()#seven independent GNN models
		self.layer_num = layer_num
		self.class_num = class_num
		self.feature_fusion = feature_fusion
		#self.f1_transform = 64
		self.layer_num = layer_num
		self.in_len = in_len
		self.input_dim = input_dim
		#self.long_conv = hyena.HyenaOperator(d_model=input_dim,l_max=in_len)#
		#self.fc1 = nn.Linear(math.floor( in_len / pool_size),self.f1_transform )
		self.hyper_dim = int(self.input_dim)
		self.mainfold_name = args.mainfold
		self.mainfold = getattr(mainfold,self.mainfold_name)()
		self.feature_fusion = feature_fusion
		self.layer_num = layer_num

		dims = [self.input_dim] + ([self.hyper_dim] * (layer_num))

		if self.mainfold.name == 'Hyperboloid':
			dims[0] += 1		
		n_curvatures = len(dims)
		self.radius = radius
		if radius is None:
			self.curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
		else:
			self.curvatures = [torch.tensor([radius]) for _ in range(n_curvatures)]         # fixed curvature
		self.curvatures.append(self.radius)

		act = getattr(torch.nn.functional, act)
		acts = [act] * (layer_num)
		for c in range(class_num):
			graph_layers = []
			for i in range(layer_num-1):
				c_in, c_out = self.curvatures[0], self.curvatures[1]
				in_dim, out_dim = dims[i], dims[i+1]
				graph_layers.append(HyperbolicGCN(self.mainfold,in_dim,out_dim,c_in, c_out,dropout,acts[i],bias,use_att,local_agg))
			in_dim, out_dim = dims[-2], dims[-1]
			graph_layers.append(torch_geometric.nn.models.GIN(in_dim,out_dim,1,out_dim,act='tanh',norm=nn.BatchNorm1d(out_dim)))
			self.models.append(nn.Sequential(*graph_layers))
		
		hidden3 = dims[0]+class_num*sum(dims[1:])	

		#self.fc2 = get_classifier(hidden3,class_num,feature_fusion)
		fc2_dim = hidden3*2
		self.fc2 = nn.Sequential(
		  nn.Linear(fc2_dim,int(fc2_dim/2)),
		  nn.ReLU(),
		  nn.Linear(int(fc2_dim/2),int(fc2_dim/4)),
		  nn.ReLU(),
		  nn.Linear(int(fc2_dim/4),class_num),
		)
		return

	def forward(self,data,edge_id=None):
		f1 = data.embed1 #f1 = data.encode1
		sparse_adj = data.sparse_adj1
		edges = data.edge1
		edge_index = data.edge2
		#f1 = self.fc1(f1)

		if self.mainfold_name == 'Hyperboloid':
			o = torch.zeros_like(f1)
			f1 = torch.cat([o[:, 0:1], f1], dim=1)
		output = [f1]

		for i,m in enumerate(self.models):
			tmp = f1
			for j in range(self.layer_num-1):
				input = (tmp,sparse_adj[i])	
				tmp, _ = m[j](input)
				output.append(tmp)
			tmp =  m[-1](tmp,edges[i])
			output.append(tmp)

		x = torch.cat(output,dim=1)
		node_id = edge_index[:, edge_id]
		x1 = x[node_id[0]]
		x2 = x[node_id[1]]

		x = torch.cat([x1, x2], dim=1) #torch.mul(x1, x2)
		x = self.fc2(x)
		return x

	def forward(self,data,edge_id=None):
		f1 = data.embed1 #f1 = data.encode1
		sparse_adj = data.sparse_adj1
		edges = data.edge1
		edge_index = data.edge2
		#f1 = self.fc1(f1)

		if self.mainfold_name == 'Hyperboloid':
			o = torch.zeros_like(f1)
			f1 = torch.cat([o[:, 0:1], f1], dim=1)
		output = [f1]

		for i,m in enumerate(self.models):
			tmp = f1
			for j in range(self.layer_num):
				tmp =  m[-1](tmp,edges[i])
				output.append(tmp)


		x = torch.cat(output,dim=1)
		node_id = edge_index[:, edge_id]
		x1 = x[node_id[0]]
		x2 = x[node_id[1]]

		x = torch.cat([x1, x2], dim=1) #torch.mul(x1, x2)
		x = self.fc2(x)
		return x

class GatedInteractionNetwork(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, use_degree_conditioned_gate=False, degree_feature_dim=0):
		super(GatedInteractionNetwork, self).__init__()
		self.use_degree_conditioned_gate = bool(use_degree_conditioned_gate) and int(degree_feature_dim) > 0
		self.degree_feature_dim = int(degree_feature_dim)
		self.fc_interaction = nn.Linear(input_dim, hidden_dim)
		if self.use_degree_conditioned_gate:
			gate_input_dim = input_dim * 4 + self.degree_feature_dim * 2
			self.fc_gate = nn.Sequential(
				nn.Linear(gate_input_dim, hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, hidden_dim),
			)
		else:
			self.fc_gate = nn.Linear(input_dim, hidden_dim)
		self.fc_output = nn.Linear(hidden_dim, output_dim)
		
	def forward(self, x1, x2, degree_feat_u=None, degree_feat_v=None):
		interaction = x1 * x2 
		# Gating mechanism
		if self.use_degree_conditioned_gate and degree_feat_u is not None and degree_feat_v is not None:
			gate_input = torch.cat([
				x1,
				x2,
				torch.abs(x1 - x2),
				interaction,
				degree_feat_u,
				degree_feat_v,
			], dim=-1)
			gate = torch.sigmoid(self.fc_gate(gate_input))
		else:
			gate = torch.sigmoid(self.fc_gate(x1 + x2)) 
		gated_interaction = gate * F.relu(self.fc_interaction(interaction))
		output = self.fc_output(gated_interaction)
		
		return output



class FactorizedBilinearPooling(nn.Module):
	def __init__(self, input_dim1, input_dim2, output_dim, factor_dim=256):
		super(FactorizedBilinearPooling, self).__init__()
		self.W1 = nn.Linear(input_dim1, factor_dim, bias=False)
		self.W2 = nn.Linear(input_dim2, factor_dim, bias=False)
		self.fc = nn.Linear(factor_dim, output_dim)
		
	def forward(self, v1, v2):
		v1_transformed = self.W1(v1)  
		v2_transformed = self.W2(v2)  
		factorized_interaction = v1_transformed * v2_transformed 
		output = self.fc(factorized_interaction)  
		return output

class GatedBilinearPooling(nn.Module):
	def __init__(self, input_dim1, input_dim2, output_dim):
		super(GatedBilinearPooling, self).__init__()
		# Bilinear weight matrix
		self.bilinear_layer = nn.Bilinear(input_dim1, input_dim2, output_dim)
		self.gate_layer1 = nn.Linear(input_dim1, output_dim)
		self.gate_layer2 = nn.Linear(input_dim2, output_dim)
		
	def forward(self, v1, v2):

		bilinear_output = self.bilinear_layer(v1, v2)
		gate_v1 = self.gate_layer1(v1)  # Linear transformation of v1
		gate_v2 = self.gate_layer2(v2)  # Linear transformation of v2
		gate = torch.sigmoid(gate_v1 + gate_v2)
		gated_bilinear_output = bilinear_output * gate
		
		return gated_bilinear_output





class CodeBook(nn.Module):
	def __init__(self, param, data_loader):
		super(CodeBook, self).__init__()
		self.param = param
		self.Protein_Encoder = GCN_Encoder(param, data_loader)
		self.Protein_Decoder = GCN_Decoder(param)
		self.vq_layer = VectorQuantizer(param['prot_hidden_dim'], param['num_embeddings'], param['commitment_cost'])

	def forward(self, batch_graph):
		z = self.Protein_Encoder.encoding(batch_graph)
		e, e_q_loss, encoding_indices = self.vq_layer(z)

		x_recon = self.Protein_Decoder.decoding(batch_graph, e)
		recon_loss = F.mse_loss(x_recon, batch_graph.ndata['x'])

		mask = torch.bernoulli(torch.full(size=(self.param['num_embeddings'],), fill_value=self.param['mask_ratio'])).bool().to(device)
		mask_index = mask[encoding_indices]
		e[mask_index] = 0.0

		x_mask_recon = self.Protein_Decoder.decoding(batch_graph, e)


		x = F.normalize(x_mask_recon[mask_index], p=2, dim=-1, eps=1e-12)
		y = F.normalize(batch_graph.ndata['x'][mask_index], p=2, dim=-1, eps=1e-12)
		mask_loss = ((1 - (x * y).sum(dim=-1)).pow_(self.param['sce_scale']))
		
		return z, e, e_q_loss, recon_loss, mask_loss.sum() / (mask_loss.shape[0] + 1e-12)





def get_classifier(hidden_layer,class_num,feature_fusion):
	fc = None
	if feature_fusion == 'CnM':
		fc = nn.Linear(3*hidden_layer,class_num)
	elif feature_fusion == 'concat':
		fc = nn.Linear(2*hidden_layer,class_num)
	elif feature_fusion == 'mul':
		fc = nn.Linear(1*hidden_layer,class_num)
	return fc


class LorentzGraphConvolution(nn.Module):
	"""
	Lorentz/HyboNet graph convolution layer.
	"""

	def __init__(
		self,
		manifold,
		in_features,
		out_features,
		use_bias,
		dropout,
		use_att,
		local_agg,
		nonlin=None,
		nonlin_mode='legacy',
	):
		super(LorentzGraphConvolution, self).__init__()
		self.linear = LorentzLinear(
			manifold,
			in_features,
			out_features,
			use_bias,
			dropout,
			nonlin=nonlin,
			nonlin_mode=nonlin_mode,
		)
		self.agg = LorentzAgg(
			manifold,
			out_features,
			dropout,
			use_att,
			local_agg,
		)

	def forward(self, input):
		x, adj = input
		h = self.linear(x)
		h = self.agg(h, adj)
		return h, adj


class LorentzLinear(nn.Module):
	def __init__(self, manifold, in_features, out_features, bias=True, dropout=0.1, scale=10.0, fixscale=False, nonlin=None, nonlin_mode='legacy'):
		super(LorentzLinear, self).__init__()
		self.manifold = manifold
		self.nonlin = nonlin
		self.nonlin_mode = nonlin_mode
		self.in_features = in_features
		self.out_features = out_features
		self.bias = bias
		self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)
		self.dropout = nn.Dropout(dropout)
		self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.out_features)
		nn.init.uniform_(self.weight.weight, -stdv, stdv)
		with torch.no_grad():
			self.weight.weight[:, 0] = 0
		if self.bias:
			nn.init.constant_(self.weight.bias, 0)

	def _apply_tangent_nonlin(self, x):
		x_tan = self.manifold.logmap0(x, c=None)
		x_tan = x_tan.clone()
		x_tan[..., 1:] = self.nonlin(x_tan[..., 1:])
		return self.manifold.expmap0(x_tan, c=None)

	def forward(self, x):
		if self.nonlin is not None:
			if self.nonlin_mode == 'legacy':
				x = self.nonlin(x)
			elif self.nonlin_mode == 'tangent':
				x = self._apply_tangent_nonlin(x)
			elif self.nonlin_mode == 'none':
				pass
			else:
				raise ValueError(f'Unknown Lorentz nonlinearity mode: {self.nonlin_mode}')
		x = self.weight(self.dropout(x))
		x_space = x.narrow(-1, 1, x.shape[-1] - 1)
		time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
		scale = (time * time - 1) / (x_space * x_space).sum(dim=-1, keepdim=True).clamp_min(1e-8)
		return torch.cat([time, x_space * scale.sqrt()], dim=-1)


class LorentzAgg(nn.Module):
	"""
	Lorentz aggregation layer used by HyboNet.
	"""

	def __init__(
		self,
		manifold,
		in_features,
		dropout,
		use_att,
		local_agg,
	):
		super(LorentzAgg, self).__init__()
		self.manifold = manifold
		self.in_features = in_features
		self.dropout = dropout
		self.local_agg = local_agg
		self.use_att = use_att
		if self.use_att:
			self.key_linear = LorentzLinear(manifold, in_features, in_features)
			self.query_linear = LorentzLinear(manifold, in_features, in_features)
			self.bias = nn.Parameter(torch.zeros(()) + 20)
			self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

	def forward(self, x, adj):
		if self.use_att and self.local_agg:
			query = self.query_linear(x)
			key = self.key_linear(x)
			att_adj = 2 + 2 * self.manifold.cinner(query, key)
			att_adj = att_adj / self.scale + self.bias
			att_adj = torch.sigmoid(att_adj)
			att_adj = torch.mul(adj.to_dense(), att_adj)
			support_t = torch.matmul(att_adj, x)
		else:
			support_t = torch.spmm(adj, x)
		denom = (-self.manifold.inner(None, None, support_t, keepdim=True)).abs().clamp_min(1e-8).sqrt()
		return support_t / denom


class HyperbolicGCN(nn.Module):
	"""
	Hyperbolic graph convolution layer.
	"""

	def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
		super(HyperbolicGCN, self).__init__()
		self.mainfold = mainfold
		self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
		self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
		self.hyp_act = HypAct(manifold, c_in, c_out, act)

	def forward(self, input):
		x, adj = input

		h = self.linear.forward(x)
		h = self.agg.forward(h, adj)
		h = self.hyp_act.forward(h)
		output = h, adj
		return output



class HypLinear(nn.Module):
	"""
	Hyperbolic linear layer.
	"""

	def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
		super(HypLinear, self).__init__()
		self.manifold = manifold
		self.in_features = in_features
		self.out_features = out_features
		self.c = c
		self.dropout = dropout
		self.use_bias = use_bias
		self.bias = nn.Parameter(torch.Tensor(out_features))
		self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
		torch.nn.init.constant_(self.bias, 0)

	def forward(self, x):
		drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
		mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
		res = self.manifold.proj(mv, self.c)
		if self.use_bias:
			bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
			hyp_bias = self.manifold.expmap0(bias, self.c)
			hyp_bias = self.manifold.proj(hyp_bias, self.c)
			res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
			res = self.manifold.proj(res, self.c)
		return res

	def extra_repr(self):
		return 'in_features={}, out_features={}, c={}'.format(
			self.in_features, self.out_features, self.c
		)


class HypAgg(nn.Module):
	"""
	Hyperbolic aggregation layer.
	"""

	def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
		super(HypAgg, self).__init__()
		self.manifold = manifold
		self.c = c

		self.in_features = in_features
		self.dropout = dropout
		self.local_agg = local_agg
		self.use_att = use_att
		if self.use_att:
			self.att = DenseAtt(in_features, dropout)

	def forward(self, x, adj):
		x_tangent = self.manifold.logmap0(x, c=self.c)
		if self.use_att:
			if self.local_agg:
				x_local_tangent = []
				for i in range(x.size(0)):
					x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
				x_local_tangent = torch.stack(x_local_tangent, dim=0)
				adj_att = self.att(x_tangent, adj)
				att_rep = adj_att.unsqueeze(-1) * x_local_tangent
				support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
				output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
				return output
			else:
				adj_att = self.att(x_tangent, adj)
				support_t = torch.matmul(adj_att, x_tangent)
		else:
			support_t = torch.spmm(adj, x_tangent)
		output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
		return output

	def extra_repr(self):
		return 'c={}'.format(self.c)


class HypAct(nn.Module):
	"""
	Hyperbolic activation layer.
	"""

	def __init__(self, manifold, c_in, c_out, act):
		super(HypAct, self).__init__()
		self.manifold = manifold
		self.c_in = c_in
		self.c_out = c_out
		self.act = act

	def forward(self, x):
		xt = self.act(self.manifold.logmap0(x, c=self.c_in))

		xt = self.manifold.proj_tan0(xt, c=self.c_out)
		return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

	def extra_repr(self):
		return 'c_in={}, c_out={}'.format(
			self.c_in, self.c_out
		)


def get_mainfold(mainfold_name):
	if mainfold_name == 'Euclidean':
		mainfold = mainfold.Euclidean()
	elif mainfold_name == 'Hyperboloid':
		mainfold = mainfold.Hyperboloid(e)
	elif mainfold_name == 'PoincareBall':
		mainfold = mainfold.PoincareBall()
	else:
		print(f'error, unrecognzied mainfold_name {mainfold_name}')

	return mainfold

class GCN_Encoder(nn.Module):
	def __init__(self, param, data_loader):
		super(GCN_Encoder, self).__init__()        
		self.data_loader = data_loader
		self.num_layers = param['prot_num_layers']
		self.dropout = nn.Dropout(param['dropout_ratio'])
		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		self.fc = nn.ModuleList()

		self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
		self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
		self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['input_dim'], param['prot_hidden_dim']), 
											'STR_KNN' : GraphConv(param['input_dim'], param['prot_hidden_dim']), 
											'STR_DIS' : GraphConv(param['input_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

		for i in range(self.num_layers - 1):
			self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
			self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
			self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
												'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
												'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

	def forward(self, vq_layer):
		prot_embed_list = []
		for iter, batch_graph in enumerate(self.data_loader):
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			batch_graph.to(device)
			h = self.encoding(batch_graph)
			z, _, _ = vq_layer(h)
			batch_graph.ndata['h'] = torch.cat([h, z], dim=-1)
			prot_embed = dgl.mean_nodes(batch_graph, 'h').detach().cpu()
			prot_embed_list.append(prot_embed)

		return torch.cat(prot_embed_list, dim=0)

	def encoding(self, batch_graph):
		x = batch_graph.ndata['x']
		for l, layer in enumerate(self.layers):
			x = layer(batch_graph, {'amino_acid': x})
			x = self.norms[l](F.relu(self.fc[l](x['amino_acid'])))
			if l != self.num_layers - 1:
				x = self.dropout(x)

		return x
		


class GCN_Decoder(nn.Module):
	def __init__(self, param):
		super(GCN_Decoder, self).__init__()
		
		self.num_layers = param['prot_num_layers']
		self.dropout = nn.Dropout(param['dropout_ratio'])
		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		self.fc = nn.ModuleList()

		for i in range(self.num_layers - 1):
			self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
			self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
			self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
												'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
												'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

		self.fc.append(nn.Linear(param['prot_hidden_dim'], param['input_dim']))
		self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
											'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
											'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))


	def decoding(self, batch_graph, x):

		for l, layer in enumerate(self.layers):
			x = layer(batch_graph, {'amino_acid': x})
			x = self.fc[l](x['amino_acid'])

			if l != self.num_layers - 1:
				x = self.dropout(self.norms[l](F.relu(x)))
			else:
				pass

		return x

class VectorQuantizer(nn.Module):
	"""
	VQ-VAE layer: Input any tensor to be quantized. 
	Args:
		embedding_dim (int): the dimensionality of the tensors in the
		quantized space. Inputs to the modules must be in this format as well.
		num_embeddings (int): the number of vectors in the quantized space.
		commitment_cost (float): scalar which controls the weighting of the loss terms.
	"""
	def __init__(self, embedding_dim, num_embeddings, commitment_cost):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.num_embeddings = num_embeddings
		self.commitment_cost = commitment_cost
		
		# initialize embeddings
		self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
		
	def forward(self, x):    
		x = F.normalize(x, p=2, dim=-1)
		encoding_indices = self.get_code_indices(x)
		quantized = self.quantize(encoding_indices)

		q_latent_loss = F.mse_loss(quantized, x.detach())
		e_latent_loss = F.mse_loss(x, quantized.detach())
		loss = q_latent_loss + self.commitment_cost * e_latent_loss

		# Straight Through Estimator
		quantized = x + (quantized - x).detach().contiguous()

		return quantized, loss, encoding_indices
	
	def get_code_indices(self, x):

		distances = (
			torch.sum(x ** 2, dim=-1, keepdim=True) +
			torch.sum(F.normalize(self.embeddings.weight, p=2, dim=-1) ** 2, dim=1) -
			2. * torch.matmul(x, F.normalize(self.embeddings.weight.t(), p=2, dim=0))
		)
		
		encoding_indices = torch.argmin(distances, dim=1)
		
		return encoding_indices
	
	def quantize(self, encoding_indices):
		"""Returns embedding tensor for a batch of indices."""
		return F.normalize(self.embeddings(encoding_indices), p=2, dim=-1)
