from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _gaussian_bump(t: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def simulate_curvature(num_layers: int = 12, num_steps: int = 100, seed: int = 42) -> np.ndarray:
    """
    Simulate ACG-learned curvature c with two adaptive patterns:
    1) Deeper layers have larger baseline curvature.
    2) Specific complex time intervals trigger additional curvature increases.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(num_steps, dtype=np.float32)

    # "Complex" intervals on the timeline.
    complexity = (
        _gaussian_bump(t, center=22, width=6.0, amp=0.55)
        + _gaussian_bump(t, center=58, width=8.0, amp=0.80)
        + _gaussian_bump(t, center=82, width=5.0, amp=0.60)
    )
    complexity += 0.06 * np.sin(2 * np.pi * t / 25.0)  # mild periodic variation
    complexity = np.clip(complexity, 0.0, None)

    c = np.zeros((num_layers, num_steps), dtype=np.float32)
    for layer in range(num_layers):
        depth_ratio = layer / max(1, num_layers - 1)  # 0 (shallow) -> 1 (deep)
        base = 0.05 + 0.22 * depth_ratio
        # deeper layers react more strongly to complex intervals
        adaptive_gain = 0.5 + 0.9 * depth_ratio
        noise = 0.01 * rng.normal(size=num_steps).astype(np.float32)
        c[layer] = base + adaptive_gain * complexity + noise

    # Keep curvature positive and in a realistic range.
    c = np.clip(c, 0.01, None)
    return c


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    num_layers = 12
    num_steps = 100
    c = simulate_curvature(num_layers=num_layers, num_steps=num_steps, seed=42)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    # You can switch between "viridis" and "inferno".
    im = ax.imshow(
        c,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        extent=[0, num_steps - 1, 1, num_layers],
    )

    ax.set_xlabel("Sequence Time Step")
    ax.set_ylabel("Model Layer")
    ax.set_title(r"ACG-Learned Curvature Heatmap")
    ax.set_yticks(np.arange(1, num_layers + 1, 1))
    ax.set_xticks(np.arange(0, num_steps + 1, 10))

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"Learned Curvature $c$")

    plt.tight_layout()
    out_path = "assets/curvature_heatmap.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved curvature heatmap to: {out_path}")


if __name__ == "__main__":
    main()

