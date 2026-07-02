import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle


def add_round_box(ax, x, y, w, h, text, fc, ec="#666666", lw=1.2, fontsize=10, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.01", facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=("bold" if bold else "normal"))


def add_arrow(ax, x1, y1, x2, y2, color="#666666", lw=1.2, style="->"):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=10, linewidth=lw, color=color)
    ax.add_patch(arr)


def main():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Colors
    c_green = "#9fd8b6"
    c_yellow = "#f3e4b5"
    c_lav = "#efe6f3"
    c_blue = "#9b9ce3"
    c_blue2 = "#7fb0e0"
    c_gray = "#e8e8ec"

    # Input
    add_round_box(ax, 0.02, 0.68, 0.13, 0.10, "Input Sequence\n$u_t$", c_gray, fontsize=12, bold=True)

    # Curvature predictor + exp projection
    add_round_box(
        ax,
        0.20,
        0.78,
        0.26,
        0.10,
        "Curvature Predictor\\n$\\phi(u_t) \\rightarrow c_t$",
        c_green,
        fontsize=12,
        bold=True,
    )
    add_round_box(
        ax,
        0.20,
        0.64,
        0.26,
        0.11,
        "Exp Projection (to hyperbolic)\\n$\\exp_0^c(\\cdot): \\mathbb{R}^d \\rightarrow \\mathbb{H}_c^d$",
        c_green,
        fontsize=12,
        bold=True,
    )

    add_arrow(ax, 0.15, 0.74, 0.20, 0.83)
    add_arrow(ax, 0.15, 0.73, 0.20, 0.69)

    # Hyperbolic node
    node = Circle((0.54, 0.69), 0.04, facecolor="#8ecf6a", edgecolor="#6faa4f", linewidth=1.2)
    ax.add_patch(node)
    ax.text(0.54, 0.69, r"$\mathbb{H}_c^d$", ha="center", va="center", fontsize=14, fontweight="bold")

    add_arrow(ax, 0.46, 0.83, 0.50, 0.83)
    ax.text(0.51, 0.84, "dynamic curvature $c_t$", fontsize=11, va="center")
    add_arrow(ax, 0.46, 0.695, 0.50, 0.695)

    # Hyperbolic SSM cell
    add_round_box(ax, 0.63, 0.60, 0.33, 0.30, "Hyperbolic SSM Cell", c_yellow, fontsize=15, bold=True)
    inner = Rectangle((0.65, 0.67), 0.29, 0.16, facecolor="#f6ebcb", edgecolor="#bda96d", linewidth=1.0, linestyle="--")
    ax.add_patch(inner)
    ax.text(0.795, 0.80, r"State Transition in $\mathbb{H}$", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(
        0.795,
        0.75,
        r"$x_t = (\bar{A} \otimes_c x_{t-1}) \oplus_c (\bar{B} \otimes_c u_t)$",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.text(0.795, 0.705, "Mobius operations", ha="center", va="center", fontsize=11)

    pert = Rectangle((0.65, 0.62), 0.29, 0.05, facecolor="#f2c94c", edgecolor="#c49c2f", linewidth=1.0)
    ax.add_patch(pert)
    ax.text(0.795, 0.645, r"+ Nonlinear Perturbation $\sigma(\cdot)$", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.795, 0.60, r"$\bar{A}$: discretized state matrix    $\bar{B}$: discretized input matrix", ha="center", va="center", fontsize=10)

    add_arrow(ax, 0.58, 0.69, 0.63, 0.69)

    # Middle panel
    mid_panel = Rectangle((0.16, 0.28), 0.80, 0.25, facecolor=c_lav, edgecolor="none")
    ax.add_patch(mid_panel)
    ax.text(0.56, 0.515, "Hierarchical Bidirectional Scan", ha="center", va="center", fontsize=13, fontweight="bold")

    f_box = Rectangle((0.20, 0.39), 0.34, 0.08, facecolor="#f7ffff", edgecolor="#277b7b", linewidth=2.0)
    b_box = Rectangle((0.58, 0.39), 0.34, 0.08, facecolor="#fff9f3", edgecolor="#9b5f2d", linewidth=2.0)
    m_box = Rectangle((0.35, 0.30), 0.40, 0.08, facecolor="#faf7ff", edgecolor="#6f3a95", linewidth=2.0)
    ax.add_patch(f_box)
    ax.add_patch(b_box)
    ax.add_patch(m_box)

    ax.text(0.22, 0.445, "Forward Scan", fontsize=12, va="center")
    ax.text(0.22, 0.41, r"$x_f = ParallelScan(x_1, x_2, \cdots, x_T)$", fontsize=12, va="center")
    ax.text(0.86, 0.445, "Backward Scan", fontsize=12, va="center", ha="right")
    ax.text(0.86, 0.41, r"$x_b = ParallelScan(x_1, x_2, \cdots, x_T)$", fontsize=12, va="center", ha="right")
    ax.text(0.55, 0.35, "Midpoint Fusion", fontsize=12, va="center", ha="center")
    ax.text(0.55, 0.32, r"$m(x_f, x_b)$ on hyperbolic manifold", fontsize=12, va="center", ha="center")

    add_arrow(ax, 0.54, 0.60, 0.54, 0.53)
    add_arrow(ax, 0.54, 0.53, 0.37, 0.47)
    add_arrow(ax, 0.54, 0.53, 0.75, 0.47)
    add_arrow(ax, 0.37, 0.39, 0.52, 0.38)
    add_arrow(ax, 0.75, 0.39, 0.58, 0.38)

    # Output
    add_round_box(
        ax,
        0.35,
        0.14,
        0.40,
        0.10,
        "Output Projection\\n$\\log_0^c(\\cdot): \\mathbb{H}_c^d \\rightarrow \\mathbb{R}^d$\\nMap back to Euclidean space",
        c_blue,
        fontsize=12,
        bold=True,
    )
    add_round_box(ax, 0.35, 0.05, 0.40, 0.06, r"Output Sequence $y_t$", c_blue2, fontsize=13, bold=True)
    add_arrow(ax, 0.55, 0.30, 0.55, 0.24)
    add_arrow(ax, 0.55, 0.14, 0.55, 0.11)

    # Legend
    lx, ly = 0.80, 0.06
    labels = [
        (c_green, "Hyperbolic operations"),
        (c_yellow, "SSM evolution"),
        (c_lav, "Bidirectional scan"),
        (c_blue, "Output projection"),
    ]
    for i, (c, t) in enumerate(labels):
        y = ly + (3 - i) * 0.045
        ax.add_patch(Rectangle((lx, y), 0.025, 0.025, facecolor=c, edgecolor="#888888", linewidth=0.8))
        ax.text(lx + 0.045, y + 0.0125, t, va="center", fontsize=11)

    plt.tight_layout()
    out = "assets/figure_architecture_english.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
