import torch


EXP_MAX_NORM = 10.0
EPS = 1e-8


def clamp(x, min=float("-inf"), max=float("+inf")):
	return torch.clamp(x, min=min, max=max)


def sqrt(x):
	return torch.sqrt(clamp(x, min=1e-9))


def acosh(x):
	x = clamp(x, min=1.0 + EPS)
	return torch.log(x + sqrt(x * x - 1.0))


def inner(u, v, keepdim=False, dim=-1):
	d = u.size(dim) - 1
	uv = u * v
	if keepdim:
		return -uv.narrow(dim, 0, 1) + uv.narrow(dim, 1, d).sum(dim=dim, keepdim=True)
	return -uv.narrow(dim, 0, 1).squeeze(dim) + uv.narrow(dim, 1, d).sum(dim=dim, keepdim=False)


def inner0(v, keepdim=False, dim=-1):
	res = -v.narrow(dim, 0, 1)
	if not keepdim:
		res = res.squeeze(dim)
	return res


def cinner(x, y):
	x = x.clone()
	x.narrow(-1, 0, 1).mul_(-1)
	return x @ y.transpose(-1, -2)


def project(x, k, dim=-1):
	dn = x.size(dim) - 1
	right = x.narrow(dim, 1, dn)
	left = torch.sqrt(k + (right * right).sum(dim=dim, keepdim=True))
	return torch.cat((left, right), dim=dim)


def project_u(x, v, k, dim=-1):
	return v.addcmul(inner(x, v, dim=dim, keepdim=True), x / k)


def project_u0(u):
	vals = torch.zeros_like(u)
	vals[..., 0:1] = u.narrow(-1, 0, 1)
	return u - vals


def norm(u, keepdim=False, dim=-1):
	return sqrt(inner(u, u, keepdim=keepdim, dim=dim))


def dist(x, y, k, keepdim=False, dim=-1):
	d = -inner(x, y, dim=dim, keepdim=keepdim)
	return acosh(d / k)


def expmap(x, u, k, dim=-1):
	nomin = norm(u, keepdim=True, dim=dim).clamp_min(EPS)
	u = u / nomin
	nomin = nomin.clamp_max(EXP_MAX_NORM)
	return torch.cosh(nomin) * x + torch.sinh(nomin) * u


def expmap0(u, k, dim=-1):
	nomin = norm(u, keepdim=True, dim=dim).clamp_min(EPS)
	u = u / nomin
	nomin = nomin.clamp_max(EXP_MAX_NORM)
	left = torch.cosh(nomin)
	right = torch.sinh(nomin) * u
	dn = right.size(dim) - 1
	return torch.cat((left + right.narrow(dim, 0, 1), right.narrow(dim, 1, dn)), dim=dim)


def logmap(x, y, k, dim=-1):
	dist_xy = dist(x, y, k=k, dim=dim, keepdim=True)
	nomin = y + 1.0 / k * inner(x, y, keepdim=True, dim=dim) * x
	denom = norm(nomin, keepdim=True, dim=dim).clamp_min(EPS)
	return dist_xy * nomin / denom


def clogmap(x, y):
	alpha = (-cinner(x, y).unsqueeze(-1)).clamp_min(1 + 1e-6)
	nomin = acosh(alpha)
	denom = (alpha * alpha - 1).sqrt()
	return nomin / denom * (y.unsqueeze(-3) - alpha * x.unsqueeze(-2))


def logmap0(y, k, dim=-1):
	alpha = -inner0(y, keepdim=True, dim=dim)
	zero_point = torch.zeros(y.shape[-1], device=y.device, dtype=y.dtype)
	zero_point[0] = 1
	denom = torch.sqrt((alpha * alpha - 1).clamp_min(EPS))
	return acosh(alpha) / denom * (y - alpha * zero_point)


def parallel_transport(x, y, v, k, dim=-1):
	nomin = inner(y, v, keepdim=True, dim=dim)
	denom = torch.clamp_min(k - inner(x, y, keepdim=True, dim=dim), 1e-7)
	return v.addcmul(nomin / denom, x + y)


def parallel_transport0(y, v, k, dim=-1):
	nomin = inner(y, v, keepdim=True, dim=dim)
	denom = torch.clamp_min(k - inner0(y, keepdim=True, dim=dim), 1e-7)
	zero_point = torch.zeros_like(y)
	zero_point[..., 0] = 1
	return v.addcmul(nomin / denom, y + zero_point)
