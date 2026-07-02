"""Evaluation metrics for NL-H-H-SSM."""

from __future__ import annotations

from typing import Optional

import torch

from nlh_ssm.ops import hyperbolic as H

_EPS = 1e-8


def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_tensor(0.0)
    return x.mean()


def smape(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    eps: float = _EPS,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).

    Formula:
        sMAPE = 200 * |y_pred - y_true| / (|y_pred| + |y_true| + eps)
    """
    denom = y_pred.abs() + y_true.abs()
    val = 200.0 * (y_pred - y_true).abs() / denom.clamp_min(eps)
    if reduction == "none":
        return val
    if reduction == "sum":
        return val.sum()
    return _safe_mean(val)


def rmsse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_train: torch.Tensor,
    *,
    seasonality: int = 1,
    eps: float = _EPS,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Root Mean Squared Scaled Error (M5-style).

    Scale denominator is based on in-sample naive seasonal forecast errors:
        scale = mean((y_t - y_{t-m})^2),  m = seasonality
        RMSSE = sqrt( mean((y_pred - y_true)^2) / (scale + eps) )

    Expected shapes:
    - y_pred, y_true: broadcastable, typically (..., H)
    - y_train:        (..., T)
    """
    if seasonality < 1:
        raise ValueError("seasonality must be >= 1")
    if y_train.size(-1) <= seasonality:
        raise ValueError("y_train length must be greater than seasonality")

    err2 = (y_pred - y_true) ** 2
    mse = err2.mean(dim=-1)

    diff = y_train[..., seasonality:] - y_train[..., :-seasonality]
    scale = (diff**2).mean(dim=-1).clamp_min(eps)
    score = torch.sqrt(mse / scale)

    if reduction == "none":
        return score
    if reduction == "sum":
        return score.sum()
    return _safe_mean(score)


def adaptive_curvature_divergence(
    points: torch.Tensor,
    expected_hier_dist: torch.Tensor,
    c: torch.Tensor | float,
    *,
    mask: Optional[torch.Tensor] = None,
    eps: float = _EPS,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Root-mean geodesic **distance error** vs.\ a tree prior (diagnostic / legacy).

    This is **not** the full paper ACD (relative distortion + ``exp(-c)`` weighting);
    use :func:`acd_paper` for the manuscript definition.
    """
    if points.dim() < 2:
        raise ValueError("points must have shape (..., N, D)")
    if expected_hier_dist.dim() < 2:
        raise ValueError("expected_hier_dist must have shape (..., N, N)")

    x_i = points.unsqueeze(-2)  # (..., N, 1, D)
    x_j = points.unsqueeze(-3)  # (..., 1, N, D)
    pred_dist = H.poincare_dist(x_i, x_j, c, dim=-1, keepdim=False)

    exp_dist = expected_hier_dist.to(device=pred_dist.device, dtype=pred_dist.dtype)
    diff2 = (pred_dist - exp_dist) ** 2

    if mask is None:
        # Exclude diagonal by default.
        n = points.size(-2)
        eye = torch.eye(n, device=pred_dist.device, dtype=torch.bool)
        # Broadcast eye to leading dims.
        while eye.dim() < diff2.dim():
            eye = eye.unsqueeze(0)
        valid = ~eye
    else:
        valid = mask.to(device=pred_dist.device, dtype=torch.bool)

    valid_f = valid.to(diff2.dtype)
    num = (diff2 * valid_f).sum(dim=(-2, -1))
    den = valid_f.sum(dim=(-2, -1)).clamp_min(eps)
    acd = torch.sqrt(num / den)

    if reduction == "none":
        return acd
    if reduction == "sum":
        return acd.sum()
    return _safe_mean(acd)


def acd_paper(
    points: torch.Tensor,
    expected_hier_dist: torch.Tensor,
    c: torch.Tensor | float,
    *,
    mask: Optional[torch.Tensor] = None,
    eps: float = _EPS,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    Adaptive Curvature Distortion (ACD) — manuscript-style edge average.

    For each valid pair :math:`(u,v)` with tree distance :math:`\mathcal{D}_T(u,v)>0`:

    .. math::

        \mathrm{ACD} = \frac{1}{|\mathcal{E}|}\sum_{(u,v)}
        \left| \frac{d_{\mathbb{H}_c}(z_u,z_v) - \mathcal{D}_T(u,v)}
        {\mathcal{D}_T(u,v) + \varepsilon} \right| \exp(-c_{\mathrm{eff}})

    Here ``c`` is a scalar effective curvature (or the mean of a per-step tensor).
    Pairs with ``expected_hier_dist == 0`` are excluded from the average (optional mask).
    """
    if points.dim() < 2:
        raise ValueError("points must have shape (..., N, D)")
    if expected_hier_dist.dim() < 2:
        raise ValueError("expected_hier_dist must have shape (..., N, N)")

    x_i = points.unsqueeze(-2)
    x_j = points.unsqueeze(-3)
    pred_dist = H.poincare_dist(x_i, x_j, c, dim=-1, keepdim=False)
    exp_dist = expected_hier_dist.to(device=pred_dist.device, dtype=pred_dist.dtype)

    if mask is None:
        n = points.size(-2)
        eye = torch.eye(n, device=pred_dist.device, dtype=torch.bool)
        while eye.dim() < pred_dist.dim():
            eye = eye.unsqueeze(0)
        valid = ~eye & (exp_dist.abs() > eps)
    else:
        valid = mask.to(device=pred_dist.device, dtype=torch.bool) & (exp_dist.abs() > eps)

    c_t = torch.as_tensor(c, device=pred_dist.device, dtype=pred_dist.dtype)
    if c_t.numel() > 1:
        c_eff = c_t.mean().clamp(-30.0, 30.0)
    else:
        c_eff = c_t.view(()).clamp(-30.0, 30.0)
    weight = torch.exp(-c_eff)
    rel = (pred_dist - exp_dist).abs() / (exp_dist.abs() + eps)
    term = rel * weight

    valid_f = valid.to(term.dtype)
    num = (term * valid_f).sum(dim=(-2, -1))
    den = valid_f.sum(dim=(-2, -1)).clamp_min(eps)
    score = num / den

    if reduction == "none":
        return score
    if reduction == "sum":
        return score.sum()
    return _safe_mean(score)


acd_geometry = adaptive_curvature_divergence

