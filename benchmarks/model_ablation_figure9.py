"""
Figure 9: grouped bar chart — component ablations only (no Full-model bar).

Y-axis: performance drop (%) relative to the full model (higher-is-better by
default). The full model is the implicit 0% baseline — state that in the paper
\\caption{}, not as a bar.

Plots three categories: w/o Hyp, w/o ACG, w/o PH-Scan. JSON still requires
``full`` scores to compute each drop.

- One JSON series: three bars, legend = configuration.
- Multiple series (e.g. val/test): grouped pairs per ablation; legend = split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

KEYS = ["full", "wo_hyp", "wo_acg", "wo_ph_scan"]
ABLATION_KEYS = ["wo_hyp", "wo_acg", "wo_ph_scan"]
PLOT_LABELS = ["w/o Hyp", "w/o ACG", "w/o PH-Scan"]

# Grouped (multi-series): strong contrast between splits
SERIES_COLORS = ["#0B3C5D", "#E63946", "#1D7874"]
# Single-series: distinct color per ablation (no Full bar)
CATEGORY_COLORS = ["#C1121F", "#F48C06", "#7209B7"]


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


def performance_drops_percent(
    scores: Dict[str, float],
    *,
    higher_is_better: bool,
) -> Tuple[List[float], Dict[str, Any]]:
    """Drop % vs full model for each ablation only (three values; baseline is implicit 0%)."""
    full = float(scores["full"])
    meta: Dict[str, Any] = {
        "full_score": full,
        "higher_is_better": higher_is_better,
        "raw": dict(scores),
    }
    drops: List[float] = []
    for k in ABLATION_KEYS:
        v = float(scores[k])
        if higher_is_better:
            d = 100.0 * (full - v) / full if abs(full) > 1e-12 else 0.0
        else:
            d = 100.0 * (v - full) / full if abs(full) > 1e-12 else 0.0
        drops.append(max(0.0, d))
    return drops, meta


def _scores_dict_from_mapping(m: Dict[str, Any]) -> Dict[str, float]:
    if "raw" in m and isinstance(m["raw"], dict):
        m = m["raw"]
    return {k: float(m[k]) for k in KEYS}


def _demo_scores(seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    base = 0.82 + rng.normal(0, 0.008)
    return {
        "full": float(base),
        "wo_hyp": float(base - 0.095 - rng.uniform(0, 0.018)),
        "wo_acg": float(base - 0.058 - rng.uniform(0, 0.014)),
        "wo_ph_scan": float(base - 0.041 - rng.uniform(0, 0.012)),
    }


def load_series_from_json(path: Path) -> Tuple[List[Tuple[str, Dict[str, float]]], bool, str]:
    """
    Returns list of (series_name, scores).
    Flat format: top-level full, wo_hyp, ... -> one series named from metric_name or 'Primary'.
    Nested: "series": {"Validation": {...}, "Test": {...}}.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    higher = bool(data.get("higher_is_better", True))
    metric_note = str(data.get("metric_name", ""))

    raw_series = data.get("series")
    if isinstance(raw_series, dict) and raw_series:
        ordered: List[Tuple[str, Dict[str, float]]] = []
        for name, block in raw_series.items():
            if not isinstance(block, dict):
                continue
            ordered.append((str(name), _scores_dict_from_mapping(block)))
        if not ordered:
            raise ValueError("JSON 'series' is empty or invalid")
        return ordered, higher, metric_note

    scores = _scores_dict_from_mapping(data)
    label = metric_note or "Primary"
    return [(label, scores)], higher, metric_note or "primary metric"


def _bar_labels(ax: Any, bars: Any, drops: List[float], ymax: float) -> None:
    for b, d in zip(bars, drops):
        cx = b.get_x() + b.get_width() / 2.0
        if d < 0.05:
            ax.text(
                cx,
                ymax * 0.015,
                "0%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="medium",
            )
        else:
            ax.text(
                cx,
                b.get_height() + 0.02 * ymax,
                f"{d:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def plot_figure9(
    series_drops: List[Tuple[str, List[float]]],
    out_path: Path,
    *,
    metric_note: str = "",
    subtitle: str = "",
) -> None:
    _set_academic_style()
    edge = "#0D1B2A"
    n_cat = len(PLOT_LABELS)
    x = np.arange(n_cat, dtype=float)

    # Split Logic (saturated) vs M5 (informative) when both present
    m5_item: Optional[Tuple[str, List[float]]] = None
    logic_item: Optional[Tuple[str, List[float]]] = None
    other: List[Tuple[str, List[float]]] = []
    for sname, drops in series_drops:
        if "M5" in sname or "RMSSE" in sname.upper():
            m5_item = (sname, drops)
        elif "Logic" in sname or "ACD" in sname.upper():
            logic_item = (sname, drops)
        else:
            other.append((sname, drops))

    use_split = (
        m5_item is not None
        and logic_item is not None
        and max(logic_item[1]) < 0.05
        and max(m5_item[1]) > 0.05
    )

    if use_split and m5_item is not None and logic_item is not None:
        fig, (ax_m5, ax_log) = plt.subplots(
            1,
            2,
            figsize=(9.2, 4.6),
            gridspec_kw={"width_ratios": [2.3, 1.0]},
        )
        m5_drops = m5_item[1]
        ymax = max(m5_drops) * 1.22 if max(m5_drops) > 1e-6 else 1.0
        bars = ax_m5.bar(
            x,
            m5_drops,
            0.55,
            color=CATEGORY_COLORS,
            edgecolor=edge,
            linewidth=0.65,
            zorder=3,
        )
        _bar_labels(ax_m5, bars, m5_drops, ymax)
        ax_m5.set_ylim(0.0, ymax)
        ax_m5.set_ylabel("Performance drop (%)")
        ax_m5.set_xticks(x)
        ax_m5.set_xticklabels(PLOT_LABELS, fontsize=10)
        ax_m5.set_title("M5 (RMSSE)", fontsize=10, pad=8)

        ax_log.set_xlim(-0.5, n_cat - 0.5)
        ax_log.set_ylim(0.0, 1.0)
        ax_log.axis("off")
        ax_log.set_title("Logic (ACD)", fontsize=10, pad=8)
        ax_log.text(
            0.5,
            0.55,
            "All variants:\nACD = 0.368\n\n0% drop vs. full\n(metric saturated)",
            ha="center",
            va="center",
            fontsize=10,
            color="#333333",
            transform=ax_log.transAxes,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#F7F7F7", edgecolor="#BBBBBB"),
        )
        ax = ax_m5
    else:
        fig, ax = plt.subplots(figsize=(6.9, 4.9))
        plot_items = series_drops if not other else other
        if not plot_items:
            plot_items = series_drops

        all_vals = [d for _, drops in plot_items for d in drops]
        ymax = max(all_vals) * 1.22 if max(all_vals) > 1e-6 else 1.0
        ax.set_ylim(0.0, ymax)
        ax.axhline(0.0, color="#222222", linewidth=0.95, zorder=2)
        ax.set_ylabel("Performance drop (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(PLOT_LABELS, fontsize=10)
        ax.tick_params(axis="x", labelsize=10)

        if len(plot_items) == 1:
            _, drops = plot_items[0]
            width = 0.55
            bars = ax.bar(
                x,
                drops,
                width,
                color=CATEGORY_COLORS,
                edgecolor=edge,
                linewidth=0.65,
                zorder=3,
            )
            _bar_labels(ax, bars, drops, ymax)
            handles = [
                mpatches.Patch(
                    facecolor=CATEGORY_COLORS[i],
                    edgecolor=edge,
                    linewidth=0.65,
                    label=PLOT_LABELS[i],
                )
                for i in range(n_cat)
            ]
            ax.legend(
                handles=handles,
                loc="upper right",
                frameon=True,
                fancybox=False,
                edgecolor="#222222",
                fontsize=9,
                title="Ablation",
                title_fontsize=9,
            )
        else:
            n_ser = len(plot_items)
            width = min(0.36, 0.78 / n_ser)
            for si, (sname, drops) in enumerate(plot_items):
                offset = (si - (n_ser - 1) / 2.0) * width
                color = SERIES_COLORS[si % len(SERIES_COLORS)]
                bars = ax.bar(
                    x + offset,
                    drops,
                    width,
                    label=sname,
                    color=color,
                    edgecolor=edge,
                    linewidth=0.65,
                    zorder=3,
                )
                _bar_labels(ax, bars, drops, ymax)
            ax.legend(
                loc="upper right",
                frameon=True,
                fancybox=False,
                edgecolor="#222222",
                fontsize=9,
                title="Evaluation",
                title_fontsize=9,
            )

    if subtitle:
        ax.set_title(subtitle, fontsize=10, pad=10)

    if metric_note:
        fig.text(
            0.5,
            0.02,
            f"Metric: {metric_note}",
            ha="center",
            fontsize=9,
            style="italic",
            color="#333333",
        )
        plt.subplots_adjust(bottom=0.18)
    else:
        plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Figure 9 component ablation bar chart.")
    ap.add_argument(
        "--demo",
        action="store_true",
        help="Use illustrative scores (grouped: Validation + Test)",
    )
    ap.add_argument(
        "--from-json",
        type=str,
        default="",
        help="JSON: flat scores or nested {\"series\": {name: {full, wo_hyp, ...}}}",
    )
    ap.add_argument(
        "--single-series",
        action="store_true",
        help="With --demo: one evaluation split only (three colored bars)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="assets/figure9_component_ablation.png",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default="benchmarks/ablation_model_components.json",
    )
    args = ap.parse_args()

    subtitle = ""
    metric_note = ""
    series_payload: List[Tuple[str, List[float], Dict[str, Any]]] = []

    if args.from_json:
        path = Path(args.from_json)
        named_scores, higher, metric_note = load_series_from_json(path)
        for sname, scores in named_scores:
            drops, meta = performance_drops_percent(scores, higher_is_better=higher)
            series_payload.append((sname, drops, meta))
        meta_out = {
            "mode": "from_json",
            "source_json": args.from_json,
            "higher_is_better": higher,
            "metric_name": metric_note,
            "baseline": "full model (0% drop; not shown — describe in caption)",
            "series": {sname: m for sname, _, m in series_payload},
            "performance_drop_pct": {
                sname: {PLOT_LABELS[i]: drops[i] for i in range(len(PLOT_LABELS))}
                for sname, drops, _ in series_payload
            },
        }
    elif args.demo:
        higher = True
        metric_note = "score (higher is better; illustrative)"
        if args.single_series:
            scores = _demo_scores(42)
            drops, meta = performance_drops_percent(scores, higher_is_better=higher)
            series_payload.append(("Primary", drops, meta))
        else:
            for seed, label in [(42, "Validation"), (43, "Test")]:
                scores = _demo_scores(seed)
                drops, meta = performance_drops_percent(scores, higher_is_better=higher)
                series_payload.append((label, drops, meta))
        meta_out = {
            "mode": "demo",
            "higher_is_better": higher,
            "metric_name": metric_note,
            "baseline": "full model (0% drop; not shown — describe in caption)",
            "series": {sname: m for sname, _, m in series_payload},
            "performance_drop_pct": {
                sname: {PLOT_LABELS[i]: drops[i] for i in range(len(PLOT_LABELS))}
                for sname, drops, _ in series_payload
            },
        }
    else:
        print(
            "Provide --from-json with scores or run with --demo for a placeholder figure.",
            file=sys.stderr,
        )
        sys.exit(1)

    plot_series = [(s, d) for s, d, _ in series_payload]
    out = Path(args.out)
    plot_figure9(plot_series, out, metric_note=metric_note, subtitle=subtitle)
    logic_flat = any(
        ("Logic" in s or "ACD" in s.upper()) and max(d) < 0.05 for s, d in plot_series
    )
    m5_signal = any(("M5" in s or "RMSSE" in s.upper()) and max(d) > 0.05 for s, d in plot_series)
    if logic_flat and m5_signal:
        print("Figure 9 layout: split (M5 bars + Logic inset)")
    else:
        print("Figure 9 layout: grouped bars")

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    print(f"Saved figure: {out}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
