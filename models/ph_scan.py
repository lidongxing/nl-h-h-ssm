"""Parallel hyperbolic scan (Triton / PyTorch) — facade over ``csrc``."""

from __future__ import annotations

from csrc.ph_scan_kernel import (
    _HAS_TRITON,
    ph_scan_fused_acg,
    ph_scan_reference,
    ph_scan_tangent_parallel,
)

__all__ = [
    "_HAS_TRITON",
    "ph_scan_fused_acg",
    "ph_scan_reference",
    "ph_scan_tangent_parallel",
]
