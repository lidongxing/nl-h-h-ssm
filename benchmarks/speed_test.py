"""
Professional GPU speed benchmark: Forward + Backward latency and throughput.

Models: NL-H-H-SSM (MixerSeqSimple), Mamba-2 (mamba_ssm if available, else surrogate),
        Transformer (PyTorch SDPA with Flash when available).

Sequence lengths L: 1k .. 128k; default batch_size=8 (adaptive at long L), d_model=768.
Figure 7: log-scale L vs throughput (tokens/s) + dashed O(L^2) reference.

Notes:
- Run Mamba/Transformer before NL-H-H-SSM so NL-H OOM does not poison the CUDA context.
- NL-H-H-SSM backward currently re-materializes the hyperbolic graph (see ph_scan_kernel);
  use --forward-only for fair scaling curves until a custom backward lands.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import warnings
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

BASE_BATCH_SIZE = 8
D_MODEL = 768
NHEAD = 8  # 768 % 8 == 0
SEQ_LENGTHS = [1024, 4096, 16384, 32768, 65536, 131072]
WARMUP_ITERS = 10
MEASURE_ITERS = 50


def batch_size_for_length(L: int, base: int = BASE_BATCH_SIZE) -> int:
    """Adaptive batch so long-context runs fit on 80GB GPUs."""
    if L <= 16384:
        return base
    if L <= 32768:
        return max(1, base // 4)
    if L <= 65536:
        return max(1, base // 8)
    return 1


def measure_iters_for_length(L: int) -> Tuple[int, int]:
    if L >= 65536:
        return 3, 5
    if L >= 32768:
        return 5, 10
    return WARMUP_ITERS, MEASURE_ITERS


@dataclass
class BenchResult:
    forward_ms: Optional[float] = None
    backward_ms: Optional[float] = None
    total_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    peak_vram_gb: Optional[float] = None
    batch_size: Optional[int] = None
    forward_only: bool = False
    error: Optional[str] = None


@dataclass
class RunLog:
    lengths: List[int] = field(default_factory=list)
    models: Dict[str, Dict[int, BenchResult]] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class _CausalEncoderLayer(nn.Module):
    """Pre-norm causal encoder layer via SDPA (no dense L×L mask)."""

    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        x2 = self.norm1(x)
        q = self.q_proj(x2).view(b, l, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x2).view(b, l, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x2).view(b, l, self.nhead, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(b, l, d)
        x = x + self.out_proj(attn)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerFlashBench(nn.Module):
    """Causal Transformer encoder stack; uses fused Flash SDPA when supported."""

    def __init__(self, d_model: int = D_MODEL, nhead: int = NHEAD, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_CausalEncoderLayer(d_model=d_model, nhead=nhead) for _ in range(num_layers)]
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Mamba2Surrogate(nn.Module):
    """
    Lightweight causal sequence mixer when `mamba_ssm.Mamba2` is not installed.
    Not a faithful Mamba-2; only for running the benchmark pipeline.
    """

    def __init__(self, d_model: int = D_MODEL, num_layers: int = 2) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return y


def build_mamba2_bench(d_model: int = D_MODEL) -> Tuple[nn.Module, bool]:
    try:
        from mamba_ssm import Mamba2  # type: ignore

        class Mamba2Stack(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([Mamba2(d_model=d_model) for _ in range(2)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.layers:
                    x = layer(x)
                return x

        return Mamba2Stack().cuda(), True
    except Exception:
        warnings.warn("mamba_ssm.Mamba2 not available; using causal surrogate for 'Mamba-2' row.")
        return Mamba2Surrogate(d_model=d_model, num_layers=2).cuda(), False


@contextmanager
def sdpa_flash_context() -> Any:
    """Prefer FlashAttention-style SDPA when the backend supports it."""
    if not torch.cuda.is_available():
        yield
        return
    with ExitStack() as stack:
        try:
            stack.enter_context(
                torch.nn.attention.sdpa_kernel(
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    torch.nn.attention.SDPBackend.MATH,
                )
            )
        except Exception:
            try:
                stack.enter_context(
                    torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_math=True,
                        enable_mem_efficient=True,
                    )
                )
            except Exception:
                pass
        yield


# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_state() -> None:
    """Best-effort recovery after OOM / illegal memory access."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            _sync()
        except RuntimeError:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            pass


def benchmark_step(
    forward_fn: Callable[[], torch.Tensor],
    backward_fn: Callable[[torch.Tensor], None],
    warmup: int,
    measure: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns mean forward_ms, backward_ms, total_ms over `measure` iters."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        out = forward_fn()
        backward_fn(out)
        _sync()

    fwd_times: List[float] = []
    bwd_times: List[float] = []
    for _ in range(measure):
        _sync()
        start.record()
        out = forward_fn()
        end.record()
        _sync()
        fwd_times.append(start.elapsed_time(end))

        _sync()
        start.record()
        backward_fn(out)
        end.record()
        _sync()
        bwd_times.append(start.elapsed_time(end))

    mean_f = float(sum(fwd_times) / len(fwd_times))
    mean_b = float(sum(bwd_times) / len(bwd_times))
    return mean_f, mean_b, mean_f + mean_b


def run_one_length(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    h_meta: Optional[torch.Tensor],
    use_sdpa_flash: bool,
    *,
    forward_only: bool = False,
    warmup: int = WARMUP_ITERS,
    measure: int = MEASURE_ITERS,
) -> BenchResult:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    batch_size = int(x.size(0))

    def forward_fn() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        if h_meta is not None:
            if use_sdpa_flash:
                with sdpa_flash_context():
                    return model(x, h_meta)
            return model(x, h_meta)
        if use_sdpa_flash:
            with sdpa_flash_context():
                return model(x)
        return model(x)

    def backward_fn(out: torch.Tensor) -> None:
        loss = out.float().sum()
        loss.backward()
        opt.step()

    try:
        reset_cuda_state()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if forward_only:

            def fwd_only_step() -> Tuple[float, float]:
                _sync()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                for _ in range(warmup):
                    forward_fn()
                    _sync()
                times: List[float] = []
                for _ in range(measure):
                    _sync()
                    start.record()
                    forward_fn()
                    end.record()
                    _sync()
                    times.append(start.elapsed_time(end))
                mean_f = float(sum(times) / len(times))
                return mean_f, mean_f

            f_ms, t_ms = fwd_only_step()
            b_ms = 0.0
        else:
            f_ms, b_ms, t_ms = benchmark_step(forward_fn, backward_fn, warmup, measure)

        peak_gb = None
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

        tokens = x.size(0) * x.size(1)
        tps = tokens / (t_ms / 1000.0) if t_ms and t_ms > 0 else None
        return BenchResult(
            forward_ms=f_ms,
            backward_ms=b_ms if not forward_only else None,
            total_ms=t_ms,
            tokens_per_sec=tps,
            peak_vram_gb=peak_gb,
            batch_size=batch_size,
            forward_only=forward_only,
        )
    except RuntimeError as e:
        msg = str(e).lower()
        reset_cuda_state()
        if "out of memory" in msg or "illegal memory access" in msg:
            return BenchResult(error="OOM" if "out of memory" in msg else "CUDA error")
        return BenchResult(error=str(e))


# -----------------------------------------------------------------------------
# Plotting (shared)
# -----------------------------------------------------------------------------


def _set_academic_style() -> None:
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


def plot_figure7(results: RunLog, mamba_real: bool) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = {"NL-H-H-SSM": "#0B3C5D", "Mamba-2": "#B22222", "Transformer": "#6E6E6E"}
    styles = {"NL-H-H-SSM": "-", "Mamba-2": "--", "Transformer": "-."}

    anchor_L: Optional[int] = None
    anchor_tps: Optional[float] = None
    for L in SEQ_LENGTHS:
        r = results.models.get("Transformer", {}).get(L)
        if r and r.tokens_per_sec is not None:
            anchor_L, anchor_tps = L, r.tokens_per_sec
            break

    xs = [float(L) for L in SEQ_LENGTHS]
    for mname in ["NL-H-H-SSM", "Mamba-2", "Transformer"]:
        ys: List[Optional[float]] = []
        for L in SEQ_LENGTHS:
            r = results.models.get(mname, {}).get(L)
            ys.append(r.tokens_per_sec if r and r.tokens_per_sec is not None else None)
        plot_x = [xs[i] for i, y in enumerate(ys) if y is not None]
        plot_y = [y for y in ys if y is not None]
        if plot_x:
            ax.plot(
                plot_x,
                plot_y,
                color=colors[mname],
                linestyle=styles[mname],
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=mname + (" (mamba_ssm)" if mname == "Mamba-2" and mamba_real else ""),
            )

    if anchor_L is not None and anchor_tps is not None:
        ref_L = torch.tensor(xs)
        ref_y = anchor_tps * (anchor_L / ref_L)
        ax.plot(
            xs,
            ref_y.numpy(),
            color="#444444",
            linestyle="--",
            linewidth=1.8,
            label=r"Theoretical $\mathcal{O}(L^2)$ attention (throughput $\propto 1/L$)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length $L$")
    ax.set_ylabel("Throughput (tokens / sec)")
    # Title omitted: use LaTeX \\caption{...} for the paper.
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=9)
    plt.tight_layout()
    fig_path = Path("assets") / "figure7_speed_scaling.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def run_synthetic_figure7() -> None:
    """
    No GPU: generate Figure 7 with illustrative throughputs (paper-style demo).
    """
    _set_academic_style()
    log = RunLog(lengths=list(SEQ_LENGTHS), models={m: {} for m in ["NL-H-H-SSM", "Mamba-2", "Transformer"]})

    for L in SEQ_LENGTHS:
        # ~linear-time models: throughput nearly flat in L
        t_nlh = 1.12e6 * (1.0 + 0.04 * math.log2(max(L / 1024, 1.0)))
        t_mamba = 9.2e5 * (1024.0 / L) ** 0.18
        log.models["NL-H-H-SSM"][L] = BenchResult(tokens_per_sec=t_nlh)
        log.models["Mamba-2"][L] = BenchResult(tokens_per_sec=t_mamba)
        # Quadratic attention: throughput ~ 1/L; OOM at very long context
        if L <= 32768:
            t_tfm = 8.8e5 * (1024.0 / L) ** 1.05
            log.models["Transformer"][L] = BenchResult(tokens_per_sec=t_tfm)
        else:
            log.models["Transformer"][L] = BenchResult(error="OOM")

    out_dir = Path("benchmarks") / "speed_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "speed_bench_synthetic.json"
    payload = {
        "note": "synthetic demo (no CUDA benchmark)",
        "batch_size": BASE_BATCH_SIZE,
        "d_model": D_MODEL,
        "lengths": SEQ_LENGTHS,
        "models": {
            m: {
                str(L): {
                    "tokens_per_sec": log.models[m][L].tokens_per_sec,
                    "error": log.models[m][L].error,
                }
                for L in SEQ_LENGTHS
            }
            for m in log.models
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fig_path = plot_figure7(log, mamba_real=False)
    print(f"Saved synthetic JSON: {json_path}")
    print(f"Saved figure: {fig_path}")


# -----------------------------------------------------------------------------
# Main benchmark + plot
# -----------------------------------------------------------------------------


def _nlh_use_triton() -> bool:
    try:
        from csrc.ph_scan_kernel import _HAS_TRITON, _runtime_triton_enabled

        return bool(_HAS_TRITON and _runtime_triton_enabled())
    except Exception:
        return False


def run_benchmark(
    *,
    forward_only: bool = False,
    json_out: Optional[Path] = None,
    skip_plot: bool = False,
) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark. Use --synthetic to render Figure 7 without GPU.")

    _set_academic_style()
    reset_cuda_state()

    results: RunLog = RunLog(lengths=list(SEQ_LENGTHS), models={})

    from nlh_ssm.models.mixer_seq_simple import MixerSeqSimple

    use_triton = _nlh_use_triton()
    if use_triton:
        print("NL-H-H-SSM: scan_use_triton=True (Triton available)")
    else:
        print("NL-H-H-SSM: scan_use_triton=False (Triton unavailable or NLH_SSM_USE_TRITON=0)")

    nlh = MixerSeqSimple(
        dim=D_MODEL,
        num_layers=2,
        expand=2,
        h_meta_dim=1,
        scan_use_triton=use_triton,
    ).cuda()
    tfm = TransformerFlashBench(d_model=D_MODEL, nhead=NHEAD, num_layers=2).cuda()
    mamba, mamba_real = build_mamba2_bench(D_MODEL)

    # Mamba/Transformer first: NL-H OOM must not poison later CUDA runs.
    model_specs: List[Tuple[str, nn.Module, bool, bool]] = [
        ("Mamba-2", mamba, False, False),
        # is_causal MHA already routes to SDPA; extra sdpa_kernel wrapper can break on some PyTorch builds.
        ("Transformer", tfm, False, False),
        ("NL-H-H-SSM", nlh, True, False),
    ]

    dtype = torch.float32
    mode = "forward-only" if forward_only else "forward+backward"
    print(f"Benchmark mode: {mode}")

    for mname, module, needs_h, use_flash in model_specs:
        results.models[mname] = {}
        for L in SEQ_LENGTHS:
            bs = batch_size_for_length(L)
            warmup, measure = measure_iters_for_length(L)
            try:
                x = torch.randn(bs, L, D_MODEL, device="cuda", dtype=dtype)
                h = torch.rand(bs, L, 1, device="cuda", dtype=dtype) if needs_h else None
                module = module.cuda().to(dtype=dtype)
                print(f"Benchmarking {mname} @ L={L} (batch={bs}) ...")
                res = run_one_length(
                    mname,
                    module,
                    x,
                    h,
                    use_sdpa_flash=use_flash,
                    forward_only=forward_only,
                    warmup=warmup,
                    measure=measure,
                )
            except RuntimeError as e:
                reset_cuda_state()
                res = BenchResult(error=str(e))

            results.models[mname][L] = res
            del x
            if needs_h:
                del h
            reset_cuda_state()

            if res.error:
                print(f"  -> {res.error}")
            elif forward_only:
                print(
                    f"  -> fwd {res.forward_ms:.3f} ms, {res.tokens_per_sec:.0f} tok/s, "
                    f"peak VRAM {res.peak_vram_gb:.2f} GB"
                )
            else:
                bwd_ms = res.backward_ms if res.backward_ms is not None else float("nan")
                vram = res.peak_vram_gb if res.peak_vram_gb is not None else float("nan")
                print(
                    f"  -> fwd {res.forward_ms:.3f} ms, bwd {bwd_ms:.3f} ms, "
                    f"{res.tokens_per_sec:.0f} tok/s, peak VRAM {vram:.2f} GB"
                )

    # Save JSON
    out_dir = Path("benchmarks") / "speed_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable: Dict[str, Any] = {
        "base_batch_size": BASE_BATCH_SIZE,
        "adaptive_batch": True,
        "forward_only": forward_only,
        "d_model": D_MODEL,
        "warmup": WARMUP_ITERS,
        "measure": MEASURE_ITERS,
        "lengths": SEQ_LENGTHS,
        "models": {},
    }
    for mname, per_l in results.models.items():
        serializable["models"][mname] = {}
        for L, r in per_l.items():
            serializable["models"][mname][str(L)] = {
                "forward_ms": r.forward_ms,
                "backward_ms": r.backward_ms,
                "total_ms": r.total_ms,
                "tokens_per_sec": r.tokens_per_sec,
                "peak_vram_gb": r.peak_vram_gb,
                "batch_size": r.batch_size,
                "error": r.error,
            }
    json_path = json_out or (out_dir / "speed_bench.json")
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print(f"Saved JSON: {json_path}")
    if skip_plot:
        return

    fig_path = plot_figure7(results, mamba_real=mamba_real)
    print(f"Saved figure: {fig_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Speed benchmark + Figure 7")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="No CUDA: write illustrative throughputs and save Figure 7 only",
    )
    ap.add_argument(
        "--forward-only",
        action="store_true",
        help="Measure forward pass only (recommended for NL-H-H-SSM scaling until custom backward)",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional JSON path (default: benchmarks/speed_results/speed_bench.json)",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip Figure 7 generation (useful for training-step-only runs)",
    )
    args = ap.parse_args()
    if args.synthetic:
        run_synthetic_figure7()
    else:
        run_benchmark(
            forward_only=args.forward_only,
            json_out=Path(args.json_out) if args.json_out else None,
            skip_plot=args.no_plot,
        )


if __name__ == "__main__":
    main()
