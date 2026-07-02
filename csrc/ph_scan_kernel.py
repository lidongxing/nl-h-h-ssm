"""
PH-Scan: Triton-backed parallel hyperbolic scan with torch fallback.

Key idea:
- Hyperbolic updates are non-associative on the manifold.
- We run recurrence in tangent space (associative affine form), then map back.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import torch
import torch.nn.functional as F

from nlh_ssm.ops import hyperbolic as H

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_TRITON = False
    triton = None
    tl = None

# After a compile/runtime failure, stay on torch for the rest of the process.
_TRITON_RUNTIME_BROKEN = False
_TRITON_FALLBACK_WARNED = False


def _runtime_triton_enabled() -> bool:
    """
    Triton JIT invokes the host compiler (often gcc); missing ``stddef.h`` or bad
    ``CPATH`` breaks startup on some clusters.

    - ``NLH_SSM_USE_TRITON=0`` (or false/off): never use Triton.
    - ``NLH_SSM_USE_TRITON=1`` (or true/on): allow Triton when the model requests it.
    - Unset: same as ``1`` for backwards compatibility with explicit ``scan_use_triton=True``.
    """
    v = os.environ.get("NLH_SSM_USE_TRITON", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _effective_triton(use_triton: bool, x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> bool:
    return (
        bool(use_triton)
        and not _TRITON_RUNTIME_BROKEN
        and _runtime_triton_enabled()
        and _HAS_TRITON
        and x.is_cuda
        and alpha.is_cuda
        and beta.is_cuda
    )


def _check_bld(x: torch.Tensor) -> None:
    if x.dim() != 3:
        raise ValueError("Expected shape (B, L, D)")


def _broadcast_curvature(c_curv: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
    if isinstance(c_curv, torch.Tensor):
        c = c_curv.to(device=x.device, dtype=x.dtype)
        if c.dim() == 0:
            return c
        if c.dim() == 1:
            return c.view(1, -1, 1)
        if c.dim() == 2:
            return c.unsqueeze(-1)
        return c
    return x.new_tensor(float(c_curv))


def _acg_curvature(
    x: torch.Tensor,
    h_meta: Optional[torch.Tensor],
    c_base: torch.Tensor | float,
    curv_linear: Optional[torch.nn.Linear],
    depth_gain: Optional[torch.Tensor | float],
) -> torch.Tensor:
    if h_meta is None or curv_linear is None:
        c = torch.as_tensor(c_base, device=x.device, dtype=x.dtype)
        return F.softplus(c).view(1, 1, 1).expand(x.size(0), x.size(1), 1)
    if h_meta.dim() == 2:
        h_meta = h_meta.unsqueeze(1).expand(x.size(0), x.size(1), h_meta.size(-1))
    xh = torch.cat([x, h_meta], dim=-1)
    logits = torch.as_tensor(c_base, device=x.device, dtype=x.dtype) + curv_linear(xh)
    if depth_gain is None:
        return F.softplus(logits)
    dg = F.softplus(torch.as_tensor(depth_gain, device=x.device, dtype=x.dtype))
    depth_signal = F.relu(h_meta[..., :1])
    return F.softplus(logits + dg * depth_signal)


if _HAS_TRITON:
    @triton.jit
    def _affine_scan_chunk_kernel(
        x_ptr,
        alpha_ptr,
        beta_ptr,
        h_init_ptr,
        y_ptr,
        h_last_ptr,
        stride_bx,
        stride_tx,
        stride_dx,
        stride_ba,
        stride_ta,
        stride_da,
        stride_bb,
        stride_tb,
        stride_db,
        stride_bh,
        stride_dh,
        stride_by,
        stride_ty,
        stride_dy,
        stride_bhl,
        stride_dhl,
        L,
        D,
        t_start,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // D
        d = pid % D

        if b >= 0 and d < D:
            h = tl.load(h_init_ptr + b * stride_bh + d * stride_dh)
            for k in range(BLOCK_T):
                t = t_start + k
                if t < L:
                    x = tl.load(x_ptr + b * stride_bx + t * stride_tx + d * stride_dx)
                    a = tl.load(alpha_ptr + b * stride_ba + t * stride_ta + d * stride_da)
                    bt = tl.load(beta_ptr + b * stride_bb + t * stride_tb + d * stride_db)
                    h = a * h + bt * x
                    tl.store(y_ptr + b * stride_by + t * stride_ty + d * stride_dy, h)
            tl.store(h_last_ptr + b * stride_bhl + d * stride_dhl, h)


def _affine_scan_chunked_torch(
    x_tan: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    h0_tan: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    b, l, _ = x_tan.shape
    y = torch.empty_like(x_tan)
    h = h0_tan
    for t0 in range(0, l, chunk_size):
        t1 = min(l, t0 + chunk_size)
        for t in range(t0, t1):
            h = alpha[:, t, :] * h + beta[:, t, :] * x_tan[:, t, :]
            y[:, t, :] = h
    return y


def _affine_scan_chunked_triton(
    x_tan: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    h0_tan: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    b, l, d = x_tan.shape
    y = torch.empty_like(x_tan)
    h_init = h0_tan.contiguous()
    h_last = torch.empty_like(h_init)
    grid = (b * d,)
    for t0 in range(0, l, chunk_size):
        _affine_scan_chunk_kernel[grid](
            x_tan,
            alpha,
            beta,
            h_init,
            y,
            h_last,
            x_tan.stride(0),
            x_tan.stride(1),
            x_tan.stride(2),
            alpha.stride(0),
            alpha.stride(1),
            alpha.stride(2),
            beta.stride(0),
            beta.stride(1),
            beta.stride(2),
            h_init.stride(0),
            h_init.stride(1),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            h_last.stride(0),
            h_last.stride(1),
            l,
            d,
            t0,
            BLOCK_T=chunk_size,
            num_warps=4,
        )
        h_init = h_last
    return y


def ph_scan_step(
    h_prev: torch.Tensor,
    x_t: torch.Tensor,
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    c_curv: torch.Tensor | float,
) -> torch.Tensor:
    ax = H.mobius_scalar(a_t, x_t, c_curv)
    bh = H.mobius_scalar(b_t, h_prev, c_curv)
    return H.mobius_add(ax, bh, c_curv)


def ph_scan_reference(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    h0: Optional[torch.Tensor],
    c_curv: torch.Tensor | float,
    *,
    dim: int = 1,
) -> torch.Tensor:
    if dim != 1:
        raise ValueError("ph_scan_reference currently expects sequence dim = 1")
    _check_bld(x)
    h = x.new_zeros(x.size(0), x.size(2)) if h0 is None else h0
    outs = []
    for t in range(x.size(1)):
        if isinstance(c_curv, torch.Tensor):
            if c_curv.dim() == 3:
                c_t = c_curv[:, t, :]
            elif c_curv.dim() == 2:
                c_t = c_curv[:, t].unsqueeze(-1)
            else:
                c_t = c_curv
        else:
            c_t = c_curv
        h = ph_scan_step(h, x[:, t, :], a[:, t, :], b[:, t, :], c_t)
        outs.append(h.unsqueeze(1))
    return torch.cat(outs, dim=1)


def ph_scan_tangent_parallel(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    c_curv: torch.Tensor | float,
    *,
    h0_ball: Optional[torch.Tensor] = None,
    chunk_size: int = 128,
    use_triton: bool = False,
) -> torch.Tensor:
    """
    High-speed PH-Scan path:
    1) log_map to tangent
    2) chunked affine scan (Triton or torch)
    3) exp_map back to manifold
    """
    _check_bld(x)
    c = _broadcast_curvature(c_curv, x)
    x_ball = H.project_onto_ball(x, c)
    x_tan = H.log_map(x_ball, c, x=None, dim=-1)
    h0 = x.new_zeros(x.size(0), x.size(2)) if h0_ball is None else h0_ball
    h0_tan = H.log_map(H.project_onto_ball(h0, c[:, 0, :] if isinstance(c, torch.Tensor) and c.dim() == 3 else c), c[:, 0, :] if isinstance(c, torch.Tensor) and c.dim() == 3 else c, x=None, dim=-1)

    if _effective_triton(use_triton, x, alpha, beta):
        try:
            y_tan = _affine_scan_chunked_triton(
                x_tan.contiguous(),
                alpha.contiguous(),
                beta.contiguous(),
                h0_tan.contiguous(),
                chunk_size,
            )
        except Exception as exc:  # pragma: no cover - host compiler / Triton
            global _TRITON_RUNTIME_BROKEN, _TRITON_FALLBACK_WARNED
            _TRITON_RUNTIME_BROKEN = True
            if not _TRITON_FALLBACK_WARNED:
                _TRITON_FALLBACK_WARNED = True
                warnings.warn(
                    "Triton PH-scan failed; using torch scan for this and later steps. "
                    "Typical cause: host gcc cannot find stddef.h (fix devtoolset/CPATH). "
                    "Defaults use torch; set scan_use_triton=True only with a working toolchain. "
                    f"({type(exc).__name__}: {exc})",
                    UserWarning,
                    stacklevel=2,
                )
            y_tan = _affine_scan_chunked_torch(x_tan, alpha, beta, h0_tan, chunk_size)
    else:
        y_tan = _affine_scan_chunked_torch(x_tan, alpha, beta, h0_tan, chunk_size)

    y_ball = H.exp_map(y_tan, c, x=None, dim=-1)
    return H.project_onto_ball(y_ball, c)


class _PHScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        c_curv: torch.Tensor,
        chunk_size: int,
        use_triton: bool,
    ) -> torch.Tensor:
        ctx.chunk_size = int(chunk_size)
        ctx.use_triton = _effective_triton(bool(use_triton), x, alpha, beta)
        ctx.save_for_backward(x, alpha, beta, c_curv)
        with torch.no_grad():
            return ph_scan_tangent_parallel(
                x,
                alpha,
                beta,
                c_curv,
                chunk_size=ctx.chunk_size,
                use_triton=ctx.use_triton,
            )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x, alpha, beta, c_curv = ctx.saved_tensors
        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            a_ = alpha.detach().requires_grad_(True)
            b_ = beta.detach().requires_grad_(True)
            c_ = c_curv.detach().requires_grad_(True)
            y = ph_scan_tangent_parallel(
                x_,
                a_,
                b_,
                c_,
                chunk_size=ctx.chunk_size,
                use_triton=ctx.use_triton,
            )
            grads = torch.autograd.grad(
                y,
                (x_, a_, b_, c_),
                grad_outputs=grad_out,
                retain_graph=False,
                allow_unused=True,
            )
        return grads[0], grads[1], grads[2], grads[3], None, None


def ph_scan(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    c_curv: torch.Tensor | float,
    *,
    h0_ball: Optional[torch.Tensor] = None,
    chunk_size: int = 128,
    use_triton: bool = False,
) -> torch.Tensor:
    """
    Drop-in accelerated scan API.
    """
    _check_bld(x)
    if isinstance(c_curv, torch.Tensor):
        c = c_curv.to(device=x.device, dtype=x.dtype)
    else:
        c = x.new_tensor(float(c_curv))
    if h0_ball is not None:
        # Current autograd wrapper keeps API simple; h0 path uses direct function.
        return ph_scan_tangent_parallel(x, alpha, beta, c, h0_ball=h0_ball, chunk_size=chunk_size, use_triton=use_triton)
    return _PHScanFunction.apply(x, alpha, beta, c, int(chunk_size), bool(use_triton))


def ph_scan_fused_acg(
    x: torch.Tensor,
    h_meta: Optional[torch.Tensor],
    delta: torch.Tensor,
    a_bar: torch.Tensor,
    b_bar: torch.Tensor,
    *,
    x_acg: Optional[torch.Tensor] = None,
    c_base: torch.Tensor | float,
    curv_linear: Optional[torch.nn.Linear] = None,
    depth_gain: Optional[torch.Tensor | float] = None,
    h0_ball: Optional[torch.Tensor] = None,
    chunk_size: int = 128,
    use_triton: bool = False,
) -> torch.Tensor:
    """
    Fused interface used by model path:
    - builds curvature from ACG inputs
    - discretizes to affine scan coefficients in tangent space
    - calls PH-Scan backend
    """
    _check_bld(x)
    x_for_curv = x if x_acg is None else x_acg
    c = _acg_curvature(x_for_curv, h_meta, c_base, curv_linear, depth_gain)
    alpha = 1.0 + delta * a_bar
    beta = delta * b_bar
    return ph_scan(x, alpha, beta, c, h0_ball=h0_ball, chunk_size=chunk_size, use_triton=use_triton)


def ph_scan_linear_then_project(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    c_curv: torch.Tensor | float,
    *,
    dim: int = 1,
) -> torch.Tensor:
    if dim != 1:
        raise ValueError("ph_scan_linear_then_project currently expects sequence dim = 1")
    _check_bld(x)
    h = x.new_zeros(x.size(0), x.size(2))
    out = []
    c = _broadcast_curvature(c_curv, x)
    for t in range(x.size(1)):
        h = alpha[:, t, :] * h + beta[:, t, :] * x[:, t, :]
        h = H.project_onto_ball(h, c[:, t, :] if isinstance(c, torch.Tensor) and c.dim() == 3 else c, dim=-1)
        out.append(h.unsqueeze(1))
    return torch.cat(out, dim=1)
