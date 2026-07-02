"""Forecast accuracy metrics (sMAPE, RMSSE, TW-MSE, CRPS)."""

from __future__ import annotations

import math
from typing import Literal

import torch

from nlh_ssm.utils.metrics import rmsse as _rmsse_impl
from nlh_ssm.utils.metrics import smape as _smape_impl

_EPS = 1e-8


def smape(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    eps: float = _EPS,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    return _smape_impl(y_pred, y_true, eps=eps, reduction=reduction)


def rmsse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_train: torch.Tensor,
    *,
    seasonality: int = 1,
    eps: float = _EPS,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    return _rmsse_impl(y_pred, y_true, y_train, seasonality=seasonality, eps=eps, reduction=reduction)


def tw_mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    S: torch.Tensor | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Trace-weighted MSE for hierarchical reconciliation quality.

    Given forecast residuals e = (y_true - y_pred), this computes:
        Tr(S Var[e] S^T)

    When ``S`` is omitted, identity is used and TW-MSE reduces to trace(Var[e]).

    Expected shape for y_pred/y_true: (..., N), where **N is the number of bottom
    hierarchical series** (columns). If you pass (B, L, D) with D = model channels
    (not HTS leaves), this becomes a **cross-channel** covariance diagnostic, not
    the Wickramasuriya reconciliation objective over the summation matrix ``S``.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    if y_pred.size(-1) < 1:
        raise ValueError("last dimension (num series) must be >= 1")

    # err: (T*, N) with T* pooled time steps and N bottom series / channels.
    err = (y_true - y_pred).reshape(-1, y_pred.size(-1))  # (T*, N)
    if err.size(0) < 2:
        # Fallback: covariance undefined for single sample, use MSE proxy.
        tw = (err.pow(2)).mean(dim=0).sum()
    else:
        # torch.cov: rows = variables, columns = observations -> pass (N, T*).
        cov = torch.cov(err.T)
        if cov.ndim == 0:
            # Univariate: torch.cov returns a scalar variance, not a 2D matrix.
            cov = cov.view(1, 1)
        if S is None:
            tw = torch.trace(cov)
        else:
            S_ = S.to(device=cov.device, dtype=cov.dtype)
            tw = torch.trace(S_ @ cov @ S_.T)

    if reduction == "none":
        return tw.view(1)
    if reduction == "sum":
        return tw
    return tw


def crps_gaussian(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    eps: float = _EPS,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    CRPS for univariate Gaussian forecasts (closed form).

    CRPS(N(mu, sigma^2), y) = sigma * ( z*(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi) )
    with z = (y - mu) / sigma.
    """
    sigma = sigma.clamp_min(eps)
    z = (y_true - mu) / sigma
    sqrt_pi = math.sqrt(math.pi)
    phi = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / sqrt_pi)
    if reduction == "none":
        return crps
    if reduction == "sum":
        return crps.sum()
    return crps.mean()


def crps_ensemble(
    y_true: torch.Tensor,
    y_samples: torch.Tensor,
    *,
    sample_dim: int = -1,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    CRPS for ensemble/sample forecasts.

    Formula:
      CRPS(F, y) = E|X - y| - 0.5 E|X - X'|
    where X, X' are iid draws from forecast distribution F.
    """
    if y_samples.size(sample_dim) < 1:
        raise ValueError("y_samples must include at least one sample")

    # Move samples to the last dimension.
    if sample_dim != -1:
        y_samples = y_samples.movedim(sample_dim, -1)
    y_true_exp = y_true.unsqueeze(-1)

    term1 = (y_samples - y_true_exp).abs().mean(dim=-1)
    pairwise = (y_samples.unsqueeze(-1) - y_samples.unsqueeze(-2)).abs().mean(dim=(-1, -2))
    crps = term1 - 0.5 * pairwise

    if reduction == "none":
        return crps
    if reduction == "sum":
        return crps.sum()
    return crps.mean()
