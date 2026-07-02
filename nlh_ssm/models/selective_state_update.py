"""Selective state update wrappers around NLHBlock."""

from __future__ import annotations

import torch
import torch.nn as nn

from nlh_ssm.modules.nlh_block import NLHBlock


class SelectiveStateUpdate(nn.Module):
    """
    Wrapper module exposing sequence and single-step style APIs.
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        h_meta_dim: int = 1,
        c_base: float = 0.1,
    ) -> None:
        super().__init__()
        self.block = NLHBlock(
            dim=dim,
            expand=expand,
            h_meta_dim=h_meta_dim,
            c_base=c_base,
        )

    def forward(self, x: torch.Tensor, h_meta: torch.Tensor) -> torch.Tensor:
        return self.block(x, h_meta)

    def step(self, x_t: torch.Tensor, h_meta_t: torch.Tensor) -> torch.Tensor:
        """
        Single-timestep adapter.
        - x_t: (B, D)
        - h_meta_t: (B, Hm)
        Returns:
        - y_t: (B, D)
        """
        y = self.block(x_t.unsqueeze(1), h_meta_t.unsqueeze(1))
        return y[:, 0, :]
