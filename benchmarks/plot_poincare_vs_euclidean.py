from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class Node:
    idx: int
    parent: int
    depth: int
    theta_min: float
    theta_max: float


def build_balanced_tree(
    max_depth: int = 20,
    branching: int = 2,
    max_nodes: int = 2500,
) -> List[Node]:
    """
    Build a synthetic tree with angular sectors.
    Each node keeps an interval [theta_min, theta_max] for radial layout.
    """
    nodes: List[Node] = [Node(idx=0, parent=-1, depth=0, theta_min=0.0, theta_max=2 * np.pi)]
    frontier = [nodes[0]]
    idx = 1
    for d in range(max_depth):
        if len(nodes) >= max_nodes:
            break
        nxt: List[Node] = []
        for n in frontier:
            if len(nodes) >= max_nodes:
                break
            span = (n.theta_max - n.theta_min) / branching
            for k in range(branching):
                if len(nodes) >= max_nodes:
                    break
                t0 = n.theta_min + k * span
                t1 = n.theta_min + (k + 1) * span
                child = Node(idx=idx, parent=n.idx, depth=d + 1, theta_min=t0, theta_max=t1)
                nodes.append(child)
                nxt.append(child)
                idx += 1
        frontier = nxt
        if not frontier:
            break
    return nodes


def make_high_dim_features(nodes: List[Node], max_depth: int, seed: int = 42) -> np.ndarray:
    """
    High-dimensional synthetic features for Euclidean PCA projection.
    Encodes path/depth with noise so deep hierarchy tends to clutter after PCA.
    """
    rng = np.random.default_rng(seed)
    n = len(nodes)
    feat_dim = max_depth * 4 + 8
    x = np.zeros((n, feat_dim), dtype=np.float32)

    # Build parent lookup for path extraction.
    parent = {n.idx: n.parent for n in nodes}
    depth = {n.idx: n.depth for n in nodes}

    for node in nodes:
        cur = node.idx
        d = node.depth
        # Path signature: alternating positional channels
        path_bits = []
        while cur != -1:
            p = parent[cur]
            if p == -1:
                break
            # child parity proxy
            path_bits.append(cur % 2)
            cur = p
        path_bits = path_bits[::-1]
        for i, b in enumerate(path_bits):
            base = 4 * i
            if base + 3 < feat_dim:
                x[node.idx, base + b] = 1.0
                x[node.idx, base + 2 + b] = 0.5
        # Depth channels
        x[node.idx, -8] = d / max(1, max_depth)
        x[node.idx, -7] = (d / max(1, max_depth)) ** 2
        # Add moderate noise to highlight Euclidean overlap
        x[node.idx] += 0.18 * rng.normal(size=feat_dim)
    return x


def poincare_disk_coords(nodes: List[Node], max_depth: int) -> np.ndarray:
    """
    Radial Poincare-style embedding:
    - parent near center
    - deeper nodes near boundary
    Radius follows r = tanh(alpha * depth / max_depth).
    """
    alpha = 2.8
    coords = np.zeros((len(nodes), 2), dtype=np.float32)
    for n in nodes:
        theta = 0.5 * (n.theta_min + n.theta_max)
        r = np.tanh(alpha * (n.depth / max(1, max_depth)))
        coords[n.idx, 0] = r * np.cos(theta)
        coords[n.idx, 1] = r * np.sin(theta)
    return coords


def draw_edges(ax: plt.Axes, pts: np.ndarray, nodes: List[Node], color: str, alpha: float, lw: float) -> None:
    for n in nodes:
        if n.parent >= 0:
            x0, y0 = pts[n.parent]
            x1, y1 = pts[n.idx]
            ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=lw, zorder=1)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.unicode_minus": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    max_depth = 20
    nodes = build_balanced_tree(max_depth=max_depth, branching=2, max_nodes=2500)
    depths = np.array([n.depth for n in nodes])

    # Left: Euclidean projection via PCA
    x_hd = make_high_dim_features(nodes, max_depth=max_depth, seed=42)
    x_euc = PCA(n_components=2, random_state=42).fit_transform(x_hd)

    # Right: Poincare Disk radial arrangement
    x_poincare = poincare_disk_coords(nodes, max_depth=max_depth)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    cmap = plt.get_cmap("viridis")

    # --- Left: Euclidean PCA ---
    ax = axes[0]
    draw_edges(ax, x_euc, nodes, color="#8a8a8a", alpha=0.20, lw=0.5)
    sc1 = ax.scatter(
        x_euc[:, 0],
        x_euc[:, 1],
        c=depths,
        cmap=cmap,
        s=10,
        alpha=0.85,
        linewidths=0.0,
        zorder=2,
    )
    ax.set_title("Euclidean Projection (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25, linewidth=0.6)

    # --- Right: Poincare Disk ---
    ax = axes[1]
    draw_edges(ax, x_poincare, nodes, color="#4f4f4f", alpha=0.18, lw=0.5)
    sc2 = ax.scatter(
        x_poincare[:, 0],
        x_poincare[:, 1],
        c=depths,
        cmap=cmap,
        s=12,
        alpha=0.9,
        linewidths=0.0,
        zorder=3,
    )
    # Unit circle boundary
    circle = plt.Circle((0.0, 0.0), 1.0, color="black", fill=False, linewidth=1.2, zorder=4)
    ax.add_artist(circle)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(r"Poincar\'e Disk Projection")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid(alpha=0.25, linewidth=0.6)

    # Shared depth colorbar
    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Node Depth")

    fig.suptitle("Hierarchical Structure: Euclidean vs. Poincare Representation", y=1.02, fontsize=12)
    plt.tight_layout()
    out_path = "assets/poincare_vs_euclidean.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()

