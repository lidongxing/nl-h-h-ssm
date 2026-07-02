"""
Core NL-H block for NL-H-H-SSM.

Design goals:
- ACG (Adaptive Curvature Gating): c = Softplus(c_base + Linear([X, H])).
- Hyperbolic transition in Log-Euclidean-Exp form:
    h_t = exp_0^c( Discretize(log_0^c(h_{t-1}), Delta, A_bar, log_0^c(x_t), B_bar) )
- Mamba-like module skeleton: A_log, D, in_proj, out_proj.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from csrc.ph_scan_kernel import ph_scan, ph_scan_fused_acg, ph_scan_reference
from nlh_ssm.ops import hyperbolic as H

AblationMode = str | None  # None | "wo_hyp" | "wo_acg" | "wo_ph_scan"


def _euclidean_affine_scan(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Sequential Euclidean recurrence (no manifold maps)."""
    b, l, d = x.shape
    h = x.new_zeros(b, d)
    y = torch.empty_like(x)
    for t in range(l):
        h = alpha[:, t, :] * h + beta[:, t, :] * x[:, t, :]
        y[:, t, :] = h
    return y


class NLHBlock(nn.Module):
    """
    Hyperbolic mixer block.

    Input:
    - x: (B, L, D)
    - h_meta: hierarchy metadata, shape (B, L, Hm) or (B, Hm)

    Output:
    - y: (B, L, D)
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        h_meta_dim: int = 1,
        c_base: float = 0.1,
        delta_floor: float = 1e-4,
        scan_chunk_size: int = 128,
        scan_use_triton: bool = False,
        ablation: AblationMode = None,
        fixed_curvature: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.inner_dim = dim * expand
        self.h_meta_dim = h_meta_dim
        self.delta_floor = delta_floor
        self.scan_chunk_size = scan_chunk_size
        self.scan_use_triton = scan_use_triton
        self.ablation = ablation
        self.fixed_curvature = float(fixed_curvature)

        # Mamba-style projection scaffold: D -> 2*ED, then gated activation.
        self.in_proj = nn.Linear(dim, 2 * self.inner_dim, bias=True)
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=True)

        # Diagonal state matrix and skip term (Mamba-like parameterization).
        self.A_log = nn.Parameter(torch.zeros(self.inner_dim))
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        # Discretization helpers in tangent space.
        self.delta_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.b_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)

        # ACG: curvature from [X, H]. Softplus keeps c > 0.
        self.curv_proj = nn.Linear(dim + h_meta_dim, 1, bias=True)
        self.c_base = nn.Parameter(torch.tensor(float(c_base)))
        # Positive gain enforces "deeper hierarchy -> larger curvature" trend.
        self.depth_gain_raw = nn.Parameter(torch.tensor(0.0))

    def _prepare_h_meta(self, h_meta: torch.Tensor, b: int, l: int) -> torch.Tensor:
        if h_meta.dim() == 2:
            h_meta = h_meta.unsqueeze(1).expand(b, l, h_meta.size(-1))
        if h_meta.dim() != 3:
            raise ValueError("h_meta must be (B, Hm) or (B, L, Hm)")
        if h_meta.size(0) != b or h_meta.size(1) != l:
            raise ValueError("h_meta batch/sequence dims must match x")
        if h_meta.size(-1) != self.h_meta_dim:
            raise ValueError(f"h_meta last dim must be {self.h_meta_dim}")
        return h_meta

    def _acg_curvature(self, x: torch.Tensor, h_meta: torch.Tensor) -> torch.Tensor:
        # c = Softplus(c_base + Linear([X, H])) + positive depth contribution.
        xh = torch.cat([x, h_meta], dim=-1)
        c_logits = self.c_base + self.curv_proj(xh)
        depth_signal = F.relu(h_meta[..., :1])  # interpreted as nonnegative hierarchy depth
        depth_gain = F.softplus(self.depth_gain_raw)
        c = F.softplus(c_logits + depth_gain * depth_signal)
        return c

    def forward(self, x: torch.Tensor, h_meta: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must have shape (B, L, D)")
        b, l, d = x.shape
        if d != self.dim:
            raise ValueError(f"x last dim must be {self.dim}")

        h_meta = self._prepare_h_meta(h_meta, b, l)

        # 1) Expand D -> ED with gated activation.
        x_proj = self.in_proj(x)
        x_up, x_gate = torch.chunk(x_proj, 2, dim=-1)
        x_up = x_up * torch.sigmoid(x_gate)

        # 3) Build discretization coefficients, then call scan backend.
        delta = F.softplus(self.delta_proj(x_up)) + self.delta_floor  # (B, L, ED)
        b_bar = self.b_proj(x_up)  # (B, L, ED)
        a_bar = -torch.exp(self.A_log).view(1, 1, self.inner_dim).expand(b, l, self.inner_dim)
        alpha = 1.0 + delta * a_bar
        beta = delta * b_bar

        if self.ablation == "wo_hyp":
            h_seq = _euclidean_affine_scan(x_up, alpha, beta)
            y = self.out_proj(h_seq + self.D.view(1, 1, -1) * x_up)
            return y

        if self.ablation == "wo_acg":
            c_seq = x.new_full((b, l, 1), self.fixed_curvature, dtype=x.dtype, device=x.device)
        else:
            c_seq = self._acg_curvature(x, h_meta)

        if self.ablation == "wo_ph_scan":
            a_gate = torch.sigmoid(a_bar)
            b_gate = torch.sigmoid(b_bar)
            h_ball_seq = ph_scan_reference(x_up, a_gate, b_gate, None, c_seq)
        else:
            if self.ablation == "wo_acg":
                h_ball_seq = ph_scan(
                    x_up,
                    alpha,
                    beta,
                    c_seq,
                    chunk_size=self.scan_chunk_size,
                    use_triton=self.scan_use_triton,
                )
            else:
                h_ball_seq = ph_scan_fused_acg(
                    x=x_up,
                    h_meta=h_meta,
                    delta=delta,
                    a_bar=a_bar,
                    b_bar=b_bar,
                    x_acg=x,
                    c_base=self.c_base,
                    curv_linear=self.curv_proj,
                    depth_gain=self.depth_gain_raw,
                    h0_ball=None,
                    chunk_size=self.scan_chunk_size,
                    use_triton=self.scan_use_triton,
                )

        # Output head remains Euclidean: map manifold state back via log_0^c.
        x_ball = H.project_onto_ball(x_up, c_seq)
        x_tan_seq = H.log_map(x_ball, c_seq, x=None, dim=-1)
        y_tan_seq = H.log_map(h_ball_seq, c_seq, x=None, dim=-1) + self.D.view(1, 1, -1) * x_tan_seq
        y = self.out_proj(y_tan_seq)
        return y
