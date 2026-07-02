"""
Poincaré ball \\mathbb{D}_c^n with sectional curvature K = -c (c > 0).

Feasible set: c * ||x||^2 < 1  (equivalently ||x|| < 1/\\sqrt{c}).

Small-c branches use Taylor expansions (through 4th order in relevant products)
so limits match Euclidean geometry smoothly and autograd stays finite.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

# Numerical guards (gradient-safe)
_EPS: float = 1e-8
_ATANH_BOUND: float = 1.0 - 1e-7
# When |c| or |c|·||·||^2 is tiny, use polynomial / Taylor forms
_C_TAYLOR: float = 1e-6

Scalar = Union[torch.Tensor, float]


def _as_c(c: Scalar, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(c, torch.Tensor):
        return c.to(device=ref.device, dtype=ref.dtype)
    return ref.new_tensor(c, dtype=ref.dtype)


def _lambda_x(x: torch.Tensor, c: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
    x2 = (x * x).sum(dim=dim, keepdim=keepdim)
    den = (1.0 - c * x2).clamp_min(_EPS)
    return 2.0 / den


def _tanh_div_sqrt_c_times_a(z: torch.Tensor, sqrt_c: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Compute tanh(z) / (sqrt(c) * a) with z = sqrt(c) * a / 2 for exp_map at origin.
    Taylor in (c * a^2): 1/2 - c*a^2/24 + c^2*a^4/240 - c^3*a^6/9450 + ...
    """
    ca2 = c * (a * a)
    t4 = 0.5 - ca2 / 24.0 + (ca2 * ca2) / 240.0 - (ca2**3) / 9450.0
    exact = torch.tanh(z) / (sqrt_c * a).clamp_min(_EPS)
    use = (c.abs() < _C_TAYLOR) | (ca2.abs() < _C_TAYLOR)
    return torch.where(use, t4, exact)


def _two_atanh_div_sqrt_c(r: torch.Tensor, sqrt_c: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    (2/sqrt(c)) * atanh(sqrt(c) * r) for ||y|| along radial at origin (log_map / distance).
    Taylor in (c r^2): 2r + 2r^3 c/3 + 2r^5 c^2/5 + 2r^7 c^3/7 + ...
    """
    u = (sqrt_c * r).clamp(min=-_ATANH_BOUND, max=_ATANH_BOUND)
    inv = 1.0 / sqrt_c.clamp_min(_EPS)
    exact = 2.0 * inv * torch.atanh(u)
    cr2 = c * (r * r)
    t4 = (
        2.0 * r
        + (2.0 / 3.0) * (r**3) * c
        + (2.0 / 5.0) * (r**5) * (c**2)
        + (2.0 / 7.0) * (r**7) * (c**3)
    )
    use = (c.abs() < _C_TAYLOR) | (cr2.abs() < _C_TAYLOR)
    return torch.where(use, t4, exact)


def project_onto_ball(x: torch.Tensor, c: Scalar, eps: float = 1e-5, dim: int = -1) -> torch.Tensor:
    """
    Project onto a safe interior of \\mathbb{D}_c^n: enforce c * ||x||^2 \\leq 1 - eps.
    Equivalent to ||x|| \\leq \\sqrt{(1-eps)/c}.
    """
    c_t = _as_c(c, x)
    x_norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    max_norm = torch.sqrt((1.0 - eps) / c_t.clamp_min(_EPS))
    factor = torch.where(x_norm > max_norm, max_norm / x_norm.clamp_min(_EPS), torch.ones_like(x_norm))
    return x * factor


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: Scalar) -> torch.Tensor:
    """Möbius addition x \\oplus_c y. For c \\to 0, reduces to x + y."""
    c_t = _as_c(c, x)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c_t * xy + c_t * y2) * x + (1.0 - c_t * x2) * y
    den = (1.0 + 2.0 * c_t * xy + (c_t**2) * x2 * y2).clamp_min(_EPS)
    return torch.where(c_t.abs() < _C_TAYLOR, x + y, num / den)


def mobius_scalar_mul(
    r: Union[torch.Tensor, float],
    x: torch.Tensor,
    c: Scalar,
    dim: int = -1,
) -> torch.Tensor:
    """Möbius scalar multiplication r \\otimes_c x. For c \\to 0, reduces to r * x."""
    c_t = _as_c(c, x)
    r_t = r if isinstance(r, torch.Tensor) else x.new_tensor(r)
    if isinstance(r, torch.Tensor):
        r_t = r_t.to(dtype=x.dtype, device=x.device)
        while r_t.dim() < x.dim():
            r_t = r_t.unsqueeze(-1)

    x_norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True).clamp_min(_EPS)
    sqrt_c = torch.sqrt(c_t.clamp_min(0.0))
    inner = (sqrt_c * x_norm).clamp(max=_ATANH_BOUND)
    alpha = torch.atanh(inner)
    scale = torch.tanh(r_t * alpha) / (sqrt_c * x_norm).clamp_min(_EPS)
    scale = torch.where(c_t.abs() < _C_TAYLOR, r_t, scale)
    zero = x_norm < _EPS
    return torch.where(zero.expand_as(x), torch.zeros_like(x), scale * x)


def exp_map(
    v: torch.Tensor,
    c: Scalar,
    x: Optional[torch.Tensor] = None,
    dim: int = -1,
) -> torch.Tensor:
    """
    Exponential map: T_x \\mathbb{D}_c^n \\to \\mathbb{D}_c^n.
    Default base point x = 0 (pass None).
    """
    c_t = _as_c(c, v)
    if x is None:
        v_norm = torch.linalg.vector_norm(v, dim=dim, keepdim=True)
        sqrt_c = torch.sqrt(c_t.clamp_min(0.0))
        a = v_norm.clamp_min(_EPS)
        z = sqrt_c * a * 0.5
        coef = _tanh_div_sqrt_c_times_a(z, sqrt_c, a, c_t)
        zero = v_norm < _EPS
        return torch.where(zero, torch.zeros_like(v), coef * v)

    lam = _lambda_x(x, c_t, dim=dim, keepdim=True)
    v_norm = torch.linalg.vector_norm(v, dim=dim, keepdim=True).clamp_min(_EPS)
    sqrt_c = torch.sqrt(c_t.clamp_min(0.0))
    z = sqrt_c * lam * v_norm * 0.5
    # tanh(z)/z up to z^4 for small z
    tz = torch.tanh(z) / z.clamp_min(_EPS)
    tz_taylor = 1.0 - (z * z) / 3.0 + (2.0 / 15.0) * (z**4) - (17.0 / 315.0) * (z**6)
    tz = torch.where(z.abs() < _C_TAYLOR, tz_taylor, tz)
    beta = (lam * 0.5) * tz
    zero_v = torch.linalg.vector_norm(v, dim=dim, keepdim=True) < _EPS
    tangent = torch.where(zero_v, torch.zeros_like(v), beta * v)
    return mobius_add(x, tangent, c_t)


def log_map(
    y: torch.Tensor,
    c: Scalar,
    x: Optional[torch.Tensor] = None,
    dim: int = -1,
) -> torch.Tensor:
    """
    Logarithmic map: \\mathbb{D}_c^n \\to T_x \\mathbb{D}_c^n.
    Default base point x = 0 (pass None).
    """
    c_t = _as_c(c, y)
    if x is None:
        y_norm = torch.linalg.vector_norm(y, dim=dim, keepdim=True)
        sqrt_c = torch.sqrt(c_t.clamp_min(0.0))
        a = y_norm.clamp_min(_EPS)
        mag = _two_atanh_div_sqrt_c(a, sqrt_c, c_t)
        zero = y_norm < _EPS
        return torch.where(zero, torch.zeros_like(y), mag * (y / a))

    lam = _lambda_x(x, c_t, dim=dim, keepdim=True)
    sub = mobius_add(-x, y, c_t)
    sub_norm = torch.linalg.vector_norm(sub, dim=dim, keepdim=True)
    sqrt_c = torch.sqrt(c_t.clamp_min(0.0))
    inner = (sqrt_c * sub_norm).clamp(max=_ATANH_BOUND)
    inv_sqrt_c = 1.0 / sqrt_c.clamp_min(_EPS)
    scale = (2.0 / (lam * sqrt_c).clamp_min(_EPS)) * torch.atanh(inner)
    r = sub_norm
    cr2 = c_t * (r * r)
    scale_taylor = (2.0 / lam) * (
        r
        + (r**3) * c_t / 3.0
        + (r**5) * (c_t**2) / 5.0
        + (r**7) * (c_t**3) / 7.0
    )
    use = (c_t.abs() < _C_TAYLOR) | (cr2.abs() < _C_TAYLOR)
    scale = torch.where(use, scale_taylor, scale)
    zero = sub_norm < _EPS
    direction = torch.where(zero, torch.zeros_like(sub), sub / sub_norm.clamp_min(_EPS))
    return scale * direction


def poincare_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    c: Scalar,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    Geodesic distance d(x, y) = (2/\\sqrt{c}) \\operatorname{artanh}(\\sqrt{c} \\|(-x)\\oplus_c y\\|).
    """
    c_t = _as_c(c, x)
    sub = mobius_add(-x, y, c_t)
    sub_norm = torch.linalg.vector_norm(sub, dim=dim, keepdim=keepdim)
    sqrt_c = torch.sqrt(c_t.clamp_min(0.0))
    inner = (sqrt_c * sub_norm).clamp(max=_ATANH_BOUND)
    inv_sqrt_c = 1.0 / sqrt_c.clamp_min(_EPS)
    dist = 2.0 * inv_sqrt_c * torch.atanh(inner)
    r = sub_norm
    cr2 = c_t * (r * r)
    dist_taylor = (
        2.0 * r
        + (2.0 / 3.0) * (r**3) * c_t
        + (2.0 / 5.0) * (r**5) * (c_t**2)
        + (2.0 / 7.0) * (r**7) * (c_t**3)
    )
    use = (c_t.abs() < _C_TAYLOR) | (cr2.abs() < _C_TAYLOR)
    return torch.where(use, dist_taylor, dist)


def poincare_dist_sq(
    x: torch.Tensor,
    y: torch.Tensor,
    c: Scalar,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    d = poincare_dist(x, y, c, dim=dim, keepdim=keepdim)
    return d * d


# --- Backward-compatible aliases (internal modules / tests) ---

project_to_ball = project_onto_ball


def mobius_scalar(
    r: Union[torch.Tensor, float],
    x: torch.Tensor,
    c: Scalar,
    *,
    dim: int = -1,
) -> torch.Tensor:
    return mobius_scalar_mul(r, x, c, dim=dim)


def expmap0(v: torch.Tensor, c: Scalar, *, dim: int = -1) -> torch.Tensor:
    return exp_map(v, c, x=None, dim=dim)


def logmap0(y: torch.Tensor, c: Scalar, *, dim: int = -1) -> torch.Tensor:
    return log_map(y, c, x=None, dim=dim)


def expmap(x: torch.Tensor, v: torch.Tensor, c: Scalar, *, dim: int = -1) -> torch.Tensor:
    return exp_map(v, c, x=x, dim=dim)


def logmap(x: torch.Tensor, y: torch.Tensor, c: Scalar, *, dim: int = -1) -> torch.Tensor:
    return log_map(y, c, x=x, dim=dim)
