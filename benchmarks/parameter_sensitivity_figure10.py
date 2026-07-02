"""
Figure 10: parameter sensitivity (two-panel).

Left: initial curvature c vs ACD (lower is better).
Right: capacity sweep — NL-H-H-SSM raw ACD vs Mamba-2 (raw if available, else normalized curve).

Use --from-json with output of scripts/aggregate_parameter_sensitivity.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _set_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _is_flat(values: Optional[List[float]], *, rtol: float = 1e-4, atol: float = 1e-6) -> bool:
    if not values or len(values) < 2:
        return True
    arr = np.asarray(values, dtype=float)
    return float(np.max(arr) - np.min(arr)) <= atol + rtol * max(float(np.max(np.abs(arr))), 1.0)


def _tight_ylim(ax: Any, values: List[float], *, pad_frac: float = 0.15) -> None:
    arr = np.asarray(values, dtype=float)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-9:
        mid = lo
        delta = max(0.002, abs(mid) * 0.01)
        ax.set_ylim(mid - delta, mid + delta)
    else:
        span = hi - lo
        ax.set_ylim(lo - pad_frac * span, hi + pad_frac * span)


def default_curvature_acd() -> Tuple[List[float], List[float]]:
    c_vals = [0.1, 0.5, 1.0, 1.5, 2.0]
    acd = [0.58, 0.835, 0.842, 0.828, 0.69]
    return c_vals, acd


def default_state_performance() -> Tuple[List[int], List[float], List[float]]:
    d_state = [16, 32, 64, 128]
    nlh = [0.88, 0.93, 0.97, 1.0]
    mamba = [0.87, 0.88, 0.86, 0.82]
    return d_state, nlh, mamba


def load_from_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _needs_dual_raw_axis(
    nlh_raw: List[float],
    mamba_raw: List[float],
    *,
    ratio_threshold: float = 2.0,
) -> bool:
    nlh_hi = float(np.max(np.abs(nlh_raw)))
    mamba_lo = float(np.min(mamba_raw))
    if nlh_hi < 1e-12:
        return True
    return mamba_lo / nlh_hi >= ratio_threshold


def _plot_capacity_dual_raw(
    ax1: Any,
    d_state: List[int],
    nlh_raw: List[float],
    mamba_raw: List[float],
) -> None:
    """NL-H and Mamba raw ACD on separate y-axes when magnitudes differ."""
    nlh_val = float(nlh_raw[0]) if _is_flat(nlh_raw) else float(np.mean(nlh_raw))
    ax1.axhline(
        nlh_val,
        color="#0B3C5D",
        linewidth=2.0,
        linestyle="-",
        label=f"NL-H-H-SSM (ACD={nlh_val:.3f})",
        zorder=2,
    )
    ax1.set_ylabel("NL-H-H-SSM ACD ($\downarrow$)", color="#0B3C5D")
    ax1.tick_params(axis="y", labelcolor="#0B3C5D")
    _tight_ylim(ax1, nlh_raw)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(
        d_state,
        mamba_raw,
        color="#B22222",
        linewidth=2.0,
        marker="^",
        markersize=6,
        markeredgecolor="#5C0000",
        markeredgewidth=0.6,
        linestyle="--",
        label="Mamba-2",
        zorder=3,
    )
    ax2.set_ylabel("Mamba-2 ACD ($\downarrow$)", color="#B22222")
    ax2.tick_params(axis="y", labelcolor="#B22222")
    _tight_ylim(ax2, mamba_raw)

    best_i = int(np.argmin(mamba_raw))
    ax2.annotate(
        f"best={mamba_raw[best_i]:.2f}\n($d_{{\\mathrm{{state}}}}$={d_state[best_i]})",
        xy=(d_state[best_i], mamba_raw[best_i]),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=8,
        color="#5C0000",
        arrowprops=dict(arrowstyle="->", color="#5C0000", lw=0.8),
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#222222",
        fontsize=8,
    )
    ax1.text(
        0.5,
        0.06,
        "Dual axes: NL-H saturated (left); Mamba-2 raw sweep (right)",
        transform=ax1.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )


def plot_figure10(
    c_vals: List[float],
    acd: List[float],
    d_state: List[int],
    nlh: List[float],
    mamba: List[float],
    out_path: Path,
    *,
    nlh_raw: Optional[List[float]] = None,
    mamba_raw: Optional[List[float]] = None,
    show_plateau_band: bool = False,
) -> None:
    _set_academic_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.4, 4.6))

    # --- Left: c vs ACD (lower is better) ---
    flat_c = _is_flat(acd)
    if show_plateau_band and not flat_c:
        ax0.axvspan(0.5, 1.2, facecolor="#0B3C5D", alpha=0.10, zorder=0, label="Illustrative band")

    ax0.plot(
        c_vals,
        acd,
        color="#0B3C5D",
        linewidth=2.0,
        marker="o",
        markersize=6,
        markeredgecolor="#0D1B2A",
        markeredgewidth=0.6,
        zorder=3,
    )
    ax0.set_xlabel(r"Initial curvature $c_{\mathrm{base}}$")
    ax0.set_ylabel("ACD ($\downarrow$)")
    ax0.set_xticks(c_vals)
    ax0.set_xticklabels([str(x) for x in c_vals])
    _tight_ylim(ax0, acd)
    if flat_c:
        ax0.text(
            0.5,
            0.08,
            "No measurable change\n(ACD saturated)",
            transform=ax0.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#CCCCCC"),
        )
    elif show_plateau_band:
        ax0.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#222222", fontsize=8)

    # --- Right: capacity sweep ---
    use_raw_both = (
        nlh_raw is not None
        and mamba_raw is not None
        and len(nlh_raw) == len(d_state)
        and len(mamba_raw) == len(d_state)
    )
    if use_raw_both and _needs_dual_raw_axis(nlh_raw, mamba_raw):
        _plot_capacity_dual_raw(ax1, d_state, nlh_raw, mamba_raw)
    elif use_raw_both:
        ax1.plot(
            d_state,
            nlh_raw,
            color="#0B3C5D",
            linewidth=2.0,
            marker="s",
            markersize=5.5,
            markeredgecolor="#0D1B2A",
            markeredgewidth=0.6,
            label="NL-H-H-SSM",
            zorder=3,
        )
        ax1.plot(
            d_state,
            mamba_raw,
            color="#B22222",
            linewidth=2.0,
            marker="^",
            markersize=6,
            markeredgecolor="#5C0000",
            markeredgewidth=0.6,
            linestyle="--",
            label="Mamba-2",
            zorder=3,
        )
        ax1.set_ylabel("ACD ($\downarrow$)")
        combined = list(nlh_raw) + list(mamba_raw)
        _tight_ylim(ax1, combined)
        ax1.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#222222", fontsize=9)
    elif nlh_raw is not None and len(nlh_raw) == len(d_state):
        # NL-H on left axis (raw); Mamba normalized on right when raw Mamba absent.
        ax1.axhline(
            float(nlh_raw[0]) if _is_flat(nlh_raw) else float(np.mean(nlh_raw)),
            color="#0B3C5D",
            linewidth=2.0,
            linestyle="-",
            label=(
                f"NL-H-H-SSM (ACD={nlh_raw[0]:.3f})"
                if _is_flat(nlh_raw)
                else "NL-H-H-SSM"
            ),
            zorder=2,
        )
        ax1.set_ylabel("NL-H-H-SSM ACD ($\downarrow$)")
        _tight_ylim(ax1, nlh_raw)
        ax2 = ax1.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.plot(
            d_state,
            mamba,
            color="#B22222",
            linewidth=2.0,
            marker="^",
            markersize=6,
            linestyle="--",
            label="Mamba-2 (norm.)",
            zorder=3,
        )
        ax2.set_ylabel("Mamba-2 norm. ACD (best=1)", color="#B22222")
        ax2.tick_params(axis="y", labelcolor="#B22222")
        ax2.set_ylim(-0.05, 1.08)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="#222222",
            fontsize=8,
        )
        if _is_flat(nlh_raw):
            ax1.text(
                0.5,
                0.08,
                "NL-H ACD saturated; Mamba-2 shown normalized within sweep",
                transform=ax1.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#444444",
            )
    else:
        ax1.plot(
            d_state,
            nlh,
            color="#0B3C5D",
            linewidth=2.0,
            marker="s",
            markersize=5.5,
            label="NL-H-H-SSM (norm.)",
            zorder=3,
        )
        ax2 = ax1.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.plot(
            d_state,
            mamba,
            color="#B22222",
            linewidth=2.0,
            marker="^",
            markersize=6,
            linestyle="--",
            label="Mamba-2 (norm.)",
            zorder=3,
        )
        ax2.set_ylabel("Mamba-2 relative ACD (best=1)", color="#B22222")
        ax2.tick_params(axis="y", labelcolor="#B22222")
        ax2.set_ylim(-0.05, 1.08)
        ax1.set_ylabel("NL-H-H-SSM ACD ($\downarrow$)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=8)

    ax1.set_xlabel(r"Nominal $d_{\mathrm{state}}$ (NL-H: expand$\times$16; Mamba: d_state)")
    ax1.set_xticks(d_state)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Figure 10 parameter sensitivity.")
    ap.add_argument("--demo", action="store_true", help="Illustrative demo curves")
    ap.add_argument(
        "--from-json",
        type=str,
        default="",
        help="JSON from aggregate_parameter_sensitivity.py",
    )
    ap.add_argument("--out", type=str, default="assets/figure10_parameter_sensitivity.png")
    ap.add_argument("--json-out", type=str, default="benchmarks/figure10_parameter_sensitivity.json")
    args = ap.parse_args()

    nlh_raw: Optional[List[float]] = None
    mamba_raw: Optional[List[float]] = None
    show_band = False
    input_meta: Dict[str, Any] = {}

    if args.from_json:
        path = Path(args.from_json)
        data = load_from_json(path)
        input_meta = {k: v for k, v in data.items() if k not in {"c", "acd", "d_state", "nlh_norm", "mamba_norm", "nlh_acd_raw", "mamba_acd_raw"}}
        c_vals = [float(x) for x in data["c"]]
        acd = [float(x) for x in data["acd"]]
        d_state = [int(x) for x in data["d_state"]]
        nlh = [float(x) for x in data["nlh_norm"]]
        mamba = [float(x) for x in data["mamba_norm"]]
        if "nlh_acd_raw" in data:
            nlh_raw = [float(x) for x in data["nlh_acd_raw"]]
        if "mamba_acd_raw" in data:
            mamba_raw = [float(x) for x in data["mamba_acd_raw"]]
        meta = {"mode": "from_json", "source": args.from_json, **input_meta}
    else:
        c_vals, acd = default_curvature_acd()
        d_state, nlh, mamba = default_state_performance()
        show_band = True
        meta = {"mode": "demo", "note": "Illustrative curves"}

    out = Path(args.out)
    plot_figure10(
        c_vals,
        acd,
        d_state,
        nlh,
        mamba,
        out,
        nlh_raw=nlh_raw,
        mamba_raw=mamba_raw,
        show_plateau_band=show_band,
    )

    payload = {
        **meta,
        "c": c_vals,
        "acd": acd,
        "d_state": d_state,
        "nlh_norm": nlh,
        "mamba_norm": mamba,
    }
    if nlh_raw is not None:
        payload["nlh_acd_raw"] = nlh_raw
    if mamba_raw is not None:
        payload["mamba_acd_raw"] = mamba_raw

    jp = Path(args.json_out)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved figure: {out}")
    print(f"Saved JSON: {jp}")


if __name__ == "__main__":
    main()
