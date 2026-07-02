"""Full-sequence NL-H-H-SSM stack (facade over ``nlh_ssm.models``)."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from nlh_ssm.models.mixer_seq_simple import MixerSeqSimple


def build_nlh_ssm(
    dim: int,
    *,
    num_layers: int = 2,
    expand: int = 2,
    h_meta_dim: int = 1,
    c_base: float = 0.1,
    scan_chunk_size: int = 128,
    scan_use_triton: bool = False,
) -> nn.Module:
    """Construct the default NL-H-H-SSM mixer model."""
    return MixerSeqSimple(
        dim=dim,
        num_layers=num_layers,
        expand=expand,
        h_meta_dim=h_meta_dim,
        c_base=c_base,
        scan_chunk_size=scan_chunk_size,
        scan_use_triton=scan_use_triton,
    )
