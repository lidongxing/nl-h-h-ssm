"""Distance / structure / efficiency metrics."""

from __future__ import annotations

import torch

from nlh_ssm.utils.metrics import acd_paper, adaptive_curvature_divergence

# Manuscript-style ACD (relative distortion + exp(-c)); use for paper / Table~6 expert block.
acd = acd_paper

# Legacy RMSE of geodesic distance vs.\ tree prior (diagnostic).
acd_geometry_rmse = adaptive_curvature_divergence


def mat_score(
    delta_metric: float | torch.Tensor,
    peak_vram_gb: float | torch.Tensor,
    *,
    eps: float = 1e-8,
    maximize_delta: bool = True,
) -> torch.Tensor:
    """
    Memory-Accuracy Trade-off (MAT):
        MAT = Delta(metric vs. baseline) / Peak VRAM (GB)

    ``maximize_delta=True`` means larger metric is better (e.g., improvement score).
    Set ``maximize_delta=False`` for error metrics where lower is better
    (e.g., delta = baseline_error - model_error).
    """
    delta = torch.as_tensor(delta_metric, dtype=torch.float32)
    vram = torch.as_tensor(peak_vram_gb, dtype=torch.float32).clamp_min(eps)
    signed = delta if maximize_delta else -delta
    return signed / vram


def peak_vram_gb(device: torch.device | None = None) -> float:
    """Return peak allocated VRAM in GB (0.0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    dev = device if device is not None else torch.device("cuda")
    return float(torch.cuda.max_memory_allocated(dev)) / (1024.0**3)


# Backward-compatible alias.
mat_alignment = mat_score
