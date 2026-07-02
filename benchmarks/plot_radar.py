from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _close(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values, values[:1]])


def save_radar_chart(out_path: str | Path = "assets/radar_table5_style.png") -> Path:
    # LaTeX-like typography suitable for academic figures (SCI-friendly).
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    labels = [
        r"$\mathrm{sMAPE}^{-1}$",
        r"$\mathrm{RMSSE}^{-1}$",
        r"$\mathrm{ACD}$",
        r"$\mathrm{Throughput}$",
        r"$\mathrm{VRAM\ Efficiency}$",
    ]

    # Normalized mock values in [0, 1], chosen so NL-H-H-SSM has largest area.
    # Inverted axes (sMAPE, RMSSE) are represented directly as "higher is better".
    data = {
        "NL-H-H-SSM": np.array([0.92, 0.90, 0.88, 0.84, 0.86]),
        "Mamba-2": np.array([0.84, 0.83, 0.76, 0.88, 0.80]),
        "Transformer": np.array([0.70, 0.68, 0.61, 0.62, 0.58]),
    }

    # Professional academic palette: deep blue, crimson, grey.
    colors = {
        "NL-H-H-SSM": "#0B3C5D",   # deep blue
        "Mamba-2": "#B22222",      # crimson
        "Transformer": "#6E6E6E",  # neutral grey
    }
    linestyles = {
        "NL-H-H-SSM": "-",
        "Mamba-2": "--",
        "Transformer": ":",
    }

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = _close(angles)

    fig = plt.figure(figsize=(7.4, 6.2))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#4D4D4D")
    ax.grid(color="#BFBFBF", linestyle="-", linewidth=0.7, alpha=0.7)
    ax.spines["polar"].set_color("#7A7A7A")
    ax.spines["polar"].set_linewidth(1.0)

    for model, vals in data.items():
        vals_closed = _close(vals)
        ax.plot(
            angles_closed,
            vals_closed,
            color=colors[model],
            linestyle=linestyles[model],
            linewidth=2.2,
            label=model,
        )
        if model == "NL-H-H-SSM":
            ax.fill(angles_closed, vals_closed, color=colors[model], alpha=0.16)

    ax.set_title(r"$\mathrm{Comprehensive\ Comparison\ Across\ Five\ Metrics}$", pad=22)
    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.27, 1.15), frameon=True)
    legend.get_frame().set_edgecolor("#A0A0A0")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved radar chart to: {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="assets/radar_table5_style.png")
    args = ap.parse_args()
    save_radar_chart(args.out)


if __name__ == "__main__":
    main()

