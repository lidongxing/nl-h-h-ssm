"""Hierarchy metadata → embeddings / masks (stub)."""

from __future__ import annotations

import torch
import torch.nn as nn


class HierarchyEmbedding(nn.Module):
    """Placeholder: encode tree depth / parent id into vectors."""

    def __init__(self, num_nodes: int, dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_nodes, dim)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(node_ids)
