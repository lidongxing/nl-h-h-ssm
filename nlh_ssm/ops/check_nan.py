"""Finite / NaN / Inf guards for tensors (hyperbolic + scan paths)."""

from __future__ import annotations

import torch


def assert_finite(name: str, t: torch.Tensor) -> None:
    if not torch.isfinite(t).all():
        raise RuntimeError(f"{name} has non-finite values")


def replace_nonfinite(t: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    return torch.where(torch.isfinite(t), t, t.new_tensor(fill))
