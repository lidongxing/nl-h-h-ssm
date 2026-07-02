"""Simple sequence mixer built from NLHBlock layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from nlh_ssm.modules.nlh_block import NLHBlock


class MixerSeqSimple(nn.Module):
    """
    Minimal end-to-end mixer for Step 2.5 integration.

    Input:
    - x: (B, L, D)
    - h_meta: (B, L, Hm) or (B, Hm)
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        expand: int = 2,
        h_meta_dim: int = 1,
        c_base: float = 0.1,
        scan_chunk_size: int = 128,
        scan_use_triton: bool = False,
        ablation: str | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                NLHBlock(
                    dim=dim,
                    expand=expand,
                    h_meta_dim=h_meta_dim,
                    c_base=c_base,
                    scan_chunk_size=scan_chunk_size,
                    scan_use_triton=scan_use_triton,
                    ablation=ablation,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h_meta: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = y + layer(y, h_meta)
        return self.norm(y)
