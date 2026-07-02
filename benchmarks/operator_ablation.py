"""
Operator ablation: sequential Möbius scan vs PH-Scan (tangent + Triton affine scan).

Measures kernel/wall time on CUDA with torch.cuda.Event.
Sequence lengths: 4k, 8k, 16k.
Outputs a grouped bar chart (Latency ms) and optional JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Project root on path for csrc / nlh_ssm
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csrc.ph_scan_kernel import (  # noqa: E402
    _HAS_TRITON,
    ph_scan_reference,
    ph_scan_tangent_parallel,
)

LENGTHS = [4096, 8192, 16384]
BATCH_SIZE = 4
DIM = 128
WARMUP = 5
ITERS = 10
CHUNK_SIZE = 128
SEED = 42


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def time_cuda_ms(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: List[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(sum(times) / len(times))


def build_tensors(
    device: torch.device,
    dtype: torch.dtype,
    L: int,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x, a, b (Möbius gates), alpha, beta (tangent affine gates), c (B,L,1)."""
    x = torch.randn(BATCH_SIZE, L, DIM, device=device, dtype=dtype, generator=rng) * 0.08
    a = torch.sigmoid(torch.randn(BATCH_SIZE, L, DIM, device=device, dtype=dtype, generator=rng))
    b = torch.sigmoid(torch.randn(BATCH_SIZE, L, DIM, device=device, dtype=dtype, generator=rng))
    # Stable affine coefficients for tangent scan (used by PH-Scan path)
    alpha = 0.85 + 0.1 * torch.sigmoid(torch.randn(BATCH_SIZE, L, DIM, device=device, dtype=dtype, generator=rng))
    beta = 0.05 + 0.1 * torch.sigmoid(torch.randn(BATCH_SIZE, L, DIM, device=device, dtype=dtype, generator=rng))
    c = torch.full((BATCH_SIZE, L, 1), 0.35, device=device, dtype=dtype)
    return x, a, b, alpha, beta, c


def run_ablation(
    *,
    demo: bool = False,
) -> Tuple[List[float], List[float], List[Optional[float]], dict]:
    """
    Returns vanilla_ms, phscan_ms, speedup (None if vanilla==0), meta dict.
    """
    if demo:
        # Illustrative 5–10× gap for papers / CPU-only environments
        rng = np.random.default_rng(SEED)
        vanilla = [8.0 * (L / 4096) ** 1.05 + rng.normal(0, 0.3) for L in LENGTHS]
        speedup = 6.5 + rng.uniform(-0.8, 0.8, size=len(LENGTHS))
        phscan = [max(0.5, v / s) for v, s in zip(vanilla, speedup)]
        meta = {
            "mode": "demo",
            "note": "Synthetic latencies to illustrate ~6–8× speedup; run on CUDA for measured numbers.",
            "triton_available": bool(_HAS_TRITON),
        }
        return vanilla, phscan, [v / p for v, p in zip(vanilla, phscan)], meta

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for measured ablation. Use --demo for a placeholder figure.")

    device = torch.device("cuda")
    dtype = torch.float32
    rng = torch.Generator(device=device)
    rng.manual_seed(SEED)

    vanilla_ms: List[float] = []
    phscan_ms: List[float] = []
    meta = {
        "mode": "cuda",
        "batch_size": BATCH_SIZE,
        "dim": DIM,
        "warmup": WARMUP,
        "iters": ITERS,
        "chunk_size": CHUNK_SIZE,
        "triton_available": bool(_HAS_TRITON),
        "lengths": LENGTHS,
    }

    for L in LENGTHS:
        x, a, b, alpha, beta, c = build_tensors(device, dtype, L, rng)

        def vanilla_fn() -> None:
            ph_scan_reference(x, a, b, None, c)

        def ph_fn() -> None:
            ph_scan_tangent_parallel(
                x,
                alpha,
                beta,
                c,
                h0_ball=None,
                chunk_size=CHUNK_SIZE,
                use_triton=True,
            )

        v_ms = time_cuda_ms(vanilla_fn, WARMUP, ITERS)
        p_ms = time_cuda_ms(ph_fn, WARMUP, ITERS)
        vanilla_ms.append(v_ms)
        phscan_ms.append(p_ms)

    speedups = [v / p if p > 1e-6 else None for v, p in zip(vanilla_ms, phscan_ms)]
    meta["speedups"] = speedups
    meta["vanilla_ms"] = vanilla_ms
    meta["phscan_ms"] = phscan_ms
    return vanilla_ms, phscan_ms, speedups, meta


def plot_grouped_bars(
    vanilla_ms: List[float],
    phscan_ms: List[float],
    out_path: Path,
    title_note: str = "",
) -> None:
    _set_style()
    labels = ["4k", "8k", "16k"]
    x = np.arange(len(LENGTHS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(
        x - width / 2,
        vanilla_ms,
        width,
        label="Vanilla (sequential Möbius)",
        color="#6E6E6E",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        phscan_ms,
        width,
        label="PH-Scan (tangent + Triton scan)" + (" *" if title_note else ""),
        color="#0B3C5D",
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (ms)")
    if title_note:
        ax.set_title(title_note, fontsize=10, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic latencies (~6–8× speedup) when no GPU or for slides",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="assets/figure8_operator_ablation.png",
        help="Paper Figure 8: sequential Möbius vs PH-Scan grouped bars",
    )
    ap.add_argument("--json-out", type=str, default="benchmarks/ablation_operator.json")
    args = ap.parse_args()

    vanilla_ms, phscan_ms, speedups, meta = run_ablation(demo=args.demo)

    note = ""
    if args.demo:
        note = "Demo latencies (not measured)"
    elif not _HAS_TRITON:
        note = "Triton not installed; PH-Scan used PyTorch fallback (still vs sequential Möbius)"

    # Omit figure title for LaTeX captions; optional subtitle only when needed
    plot_grouped_bars(vanilla_ms, phscan_ms, Path(args.out), title_note=note if (args.demo or not _HAS_TRITON) else "")

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved figure: {args.out}")
    print(f"Saved JSON: {args.json_out}")
    for L, v, p, s in zip(LENGTHS, vanilla_ms, phscan_ms, speedups):
        su = f"{s:.2f}×" if s is not None else "n/a"
        print(f"  L={L}: vanilla={v:.3f} ms, ph_scan={p:.3f} ms, speedup={su}")


if __name__ == "__main__":
    main()
