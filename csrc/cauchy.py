"""Cauchy / spectral helpers used by PH-Scan (weights, stability)."""

from __future__ import annotations

import torch


def cauchy_dot(
    z: torch.Tensor,
    p: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Stable Cauchy-style kernel response 1 / (z - p) along the last dim of z.

    z, p broadcast; imaginary part of z should keep denominators away from 0.
    """
    denom = z - p
    e = denom.new_tensor(eps)
    return denom / (denom.real ** 2 + denom.imag ** 2 + e).clamp_min(e)
