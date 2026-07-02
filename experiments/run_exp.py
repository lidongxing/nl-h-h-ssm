from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional, Tuple


def _setup_triton_host_compiler() -> None:
    """
    Triton 在 JIT 时用 ``$CC``（或 ``gcc``）编译很小的 host C 桩；若 ``CC`` 指向独立安装的
    gcc（例如 ``.../gcc-11/bin/gcc``），有时不会带上内置头路径，导致 ``stddef.h`` 找不到。
    仅设 ``C_INCLUDE_PATH`` 对部分 Triton 版本/调用方式不可靠，因此生成一个包装脚本作为
    ``CC``，在调用真实编译器时显式附加 ``-I<gcc-include>``。
    """
    import glob
    import hashlib
    import shlex
    import shutil
    import subprocess

    if os.name != "posix":
        return

    real_cc = os.environ.get("CC") or shutil.which("gcc")
    if not real_cc or not os.path.isfile(real_cc):
        return
    real_cc = os.path.realpath(real_cc)

    extra: list[str] = []
    for flag in ("-print-file-name=include", "-print-file-name=include-fixed"):
        try:
            r = subprocess.run(
                [real_cc, flag],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception:
            continue
        p = (r.stdout or "").strip()
        if not p:
            continue
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(os.path.dirname(real_cc), p))
        if os.path.isdir(p) and p not in extra:
            extra.append(p)

    bin_dir = os.path.dirname(real_cc)
    for pattern in (
        os.path.join(bin_dir, "..", "lib", "gcc", "*", "*", "include"),
        os.path.join(bin_dir, "..", "lib", "gcc", "*", "*", "include-fixed"),
    ):
        for p in glob.glob(os.path.normpath(pattern)):
            if os.path.isdir(p) and p not in extra:
                extra.append(p)

    conda_p = os.environ.get("CONDA_PREFIX")
    if conda_p:
        for pattern in (
            os.path.join(conda_p, "lib", "gcc", "*", "*", "include"),
            os.path.join(conda_p, "lib", "gcc", "*", "*", "include-fixed"),
        ):
            for p in glob.glob(pattern):
                if os.path.isdir(p) and p not in extra:
                    extra.append(p)

    for pattern in ("/usr/lib/gcc/*/include", "/usr/lib/gcc/*/include-fixed"):
        for p in glob.glob(pattern):
            if os.path.isdir(p) and p not in extra:
                extra.append(p)

    if not extra:
        return

    cache = Path.home() / ".cache" / "nlh_ssm_triton_cc"
    cache.mkdir(parents=True, exist_ok=True)
    key_src = real_cc + "\0" + "\0".join(extra)
    key = hashlib.sha256(key_src.encode()).hexdigest()[:20]
    wrap = cache / f"gcc-for-triton-{key}.sh"
    inc_flags = " ".join(shlex.quote("-I" + d) for d in extra)
    body = f"#!/bin/bash\nexec {shlex.quote(real_cc)} {inc_flags} \"$@\"\n"
    if not wrap.is_file() or wrap.read_text() != body:
        wrap.write_text(body)
        wrap.chmod(0o755)
    os.environ["CC"] = str(wrap)


_setup_triton_host_compiler()


def _require_triton_language_cumsum_for_mamba2() -> None:
    """
    mamba_ssm SSD Triton kernels reference ``triton.language.cumsum``; very old Triton builds omit it
    (see state-spaces/mamba issues). Fail fast with an actionable message instead of a JIT KeyError chain.
    """
    try:
        import triton.language as tl  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "使用 --model mamba2 需要已安装 Triton（一般随带 CUDA 的 PyTorch wheel 提供）。\n"
            "请先安装与 CUDA 匹配的 PyTorch，或单独安装与当前 torch 版本兼容的 triton 包。"
        ) from e
    if not hasattr(tl, "cumsum"):
        raise SystemExit(
            "当前 Triton 版本过旧或与 mamba_ssm 不兼容：`triton.language` 没有 `cumsum`，无法编译 SSD 算子。\n\n"
            "建议：\n"
            "  1) 升级 Triton（需与 PyTorch 官方说明一致），例如： pip install -U 'triton>=2.1.0'\n"
            "  2) 或按 https://pytorch.org 重装对应 CUDA 的 PyTorch（会带上匹配的 triton）。\n"
            "  3) 若出现 `ld: skipping incompatible ... libcuda.so`，请使用 64 位 Python，并设置\n"
            "     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH\n"
            "     （路径以本机 CUDA 安装为准）。\n"
        )


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from nlh_ssm.data.loader import HierarchyMeta, get_dataloader, max_series_length_long, resolve_seq_stride_long
from nlh_ssm.models.mixer_seq_simple import MixerSeqSimple
from nlh_ssm.metrics import acd, crps_gaussian, mat_score, peak_vram_gb, rmsse, smape, tw_mse


ModelName = Literal["mamba1", "mamba2", "nlh_ssm", "transformer", "informer"]


def _apply_nl_hparams_yaml(args: argparse.Namespace, dataset_stem: str) -> None:
    """Merge ``configs/nlh_tuned.yaml`` (defaults + per-dataset overrides) into ``args`` for nlh_ssm."""
    raw = getattr(args, "nlh_hparams_file", None)
    if not raw or not str(raw).strip():
        return
    import yaml

    p = Path(str(raw))
    if not p.is_file():
        print(f"[run_exp] warning: nlh_hparams_file not found: {p}")
        return
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    d = dict(cfg.get("defaults") or {})
    ov = dict((cfg.get("overrides") or {}).get(dataset_stem, {}) or {})
    merged = {**d, **ov}
    keys = ("nlh_lr", "grad_clip_norm", "nlh_num_layers", "nlh_expand", "nlh_c_base", "adamw_weight_decay", "seq_len", "stride")
    for k in keys:
        if k not in merged or merged[k] is None:
            continue
        if not hasattr(args, k):
            continue
        cur = getattr(args, k)
        v = merged[k]
        if isinstance(cur, bool):
            setattr(args, k, bool(v))
        elif isinstance(cur, int):
            setattr(args, k, int(v))
        elif isinstance(cur, float):
            setattr(args, k, float(v))
        else:
            setattr(args, k, v)
    print(f"[run_exp] merged nlh hyperparams for {dataset_stem!r} from {p}")


class _GRUBaseline(nn.Module):
    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size=dim, hidden_size=hidden, batch_first=True)
        self.proj = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, h_meta: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002
        y, _ = self.rnn(x)
        return self.proj(y)


class _TransformerBaseline(nn.Module):
    def __init__(self, dim: int, nhead: int = 4, num_layers: int = 2, ff: int = 256) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("Transformer baseline requires dim >= 1")
        # Pick the largest head count <= requested nhead that divides dim.
        nhead_eff = max(h for h in range(1, int(nhead) + 1) if dim % h == 0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead_eff,
            dim_feedforward=ff,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, h_meta: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002
        y = self.enc(x)
        return self.proj(y)


class _ProbSparseMultiheadAttention(nn.Module):
    """
    ProbSparse self-attention after Zhou et al. (Informer): keep the top-``u`` queries by
    :math:`\\bar{M}(q_i,K)=\\max_j \\frac{q_i k_j}{\\sqrt{d}} - \\frac{1}{L}\\sum_j \\frac{q_i k_j}{\\sqrt{d}}`;
    those queries use full softmax attention over keys; others use the mean of values over time.
    """

    def __init__(self, d_model: int, nhead: int, factor: float = 5.0) -> None:
        super().__init__()
        if d_model < 1 or nhead < 1 or d_model % nhead != 0:
            raise ValueError("d_model must be positive and divisible by nhead")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.dk = self.d_model // self.nhead
        self.factor = float(factor)
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, ell, _d = x.shape
        h, dk = self.nhead, self.dk
        q = self.q_proj(x).view(b, ell, h, dk).transpose(1, 2)
        k = self.k_proj(x).view(b, ell, h, dk).transpose(1, 2)
        v = self.v_proj(x).view(b, ell, h, dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (dk**0.5)
        max_s = scores.max(dim=-1).values
        mean_s = scores.mean(dim=-1)
        m_bar = max_s - mean_s
        u = max(1, min(ell, int(self.factor * math.log(max(ell, 2)))))
        topi = m_bar.topk(u, dim=-1).indices
        eligible = torch.zeros(b, h, ell, dtype=torch.bool, device=x.device)
        eligible.scatter_(2, topi, True)
        attn_out = F.softmax(scores, dim=-1) @ v
        mean_v = v.mean(dim=2, keepdim=True)
        mixed = torch.where(eligible.unsqueeze(-1), attn_out, mean_v)
        out = mixed.transpose(1, 2).contiguous().view(b, ell, self.d_model)
        return self.out_proj(out)


class _InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int, factor: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _ProbSparseMultiheadAttention(d_model, nhead, factor=factor)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class _InformerBaseline(nn.Module):
    """
    Informer-style encoder for sliding windows (B, L, D): ProbSparse attention + feed-forward,
    optional 1D conv distilling (stride 2) between stacks with linear upsample back to L.
    Matches Table 6 ``Informer`` row; not the full Informer decoder/generative path.
    """

    def __init__(
        self,
        dim: int,
        *,
        nhead: int = 4,
        ff_dim: int = 256,
        factor: float = 5.0,
        use_distill: bool = True,
        min_len_distill: int = 16,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("Informer baseline requires dim >= 1")
        nhead_eff = max(h for h in range(1, int(nhead) + 1) if dim % h == 0)
        self.use_distill = bool(use_distill)
        self.min_len_distill = int(min_len_distill)
        self.enc1 = _InformerEncoderLayer(dim, nhead_eff, ff_dim, factor)
        self.enc2 = _InformerEncoderLayer(dim, nhead_eff, ff_dim, factor)
        self.distill = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, h_meta: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002
        b, ell0, d = x.shape
        h = self.enc1(x)
        if self.use_distill and ell0 >= self.min_len_distill:
            z = self.distill(h.transpose(1, 2)).transpose(1, 2)
            h2 = self.enc2(z)
            h_up = F.interpolate(h2.transpose(1, 2), size=ell0, mode="linear", align_corners=False).transpose(1, 2)
            return self.proj(h_up)
        h2 = self.enc2(h)
        return self.proj(h2)


def _resolve_mamba_block(variant: Literal["mamba1", "mamba2"]):
    """
    Resolve Mamba block class across mamba_ssm versions.
    """
    try:
        if variant == "mamba1":
            try:
                from mamba_ssm.modules.mamba_simple import Mamba as Block  # type: ignore

                return Block
            except Exception:
                from mamba_ssm import Mamba as Block  # type: ignore

                return Block

        # mamba2 API differs across releases; try common module paths.
        try:
            from mamba_ssm.modules.mamba2_simple import Mamba2Simple as Block  # type: ignore

            _patch_mamba2_simple_forward_for_causal_length()
            return Block
        except Exception:
            pass
        try:
            from mamba_ssm.modules.mamba2 import Mamba2 as Block  # type: ignore

            return Block
        except Exception:
            pass
        from mamba_ssm import Mamba2 as Block  # type: ignore

        return Block
    except Exception as e:
        raise SystemExit(
            "Requested model requires `mamba_ssm` but it is unavailable or incompatible. "
            "Install in this env, e.g. `pip install mamba-ssm`, then retry."
        ) from e


def _force_mamba2_torch_conv1d() -> None:
    """
    Mamba2Simple (non-fused path) still calls ``causal_conv1d_fn`` when installed; that CUDA
    op requires strides multiples of 8 for some layouts and can fail on generic (B, L, D) batches.
    Clearing the hook forces the standard ``nn.Conv1d`` branch in ``mamba2_simple.forward``.
    """
    try:
        import mamba_ssm.modules.mamba2_simple as m2s  # type: ignore

        m2s.causal_conv1d_fn = None
    except Exception:
        pass
    try:
        import mamba_ssm.modules.mamba2 as m2  # type: ignore

        if hasattr(m2, "causal_conv1d_fn"):
            m2.causal_conv1d_fn = None
    except Exception:
        pass


def _patch_mamba2_simple_forward_for_causal_length() -> None:
    """
    Upstream Mamba2Simple (non-fused path) slices ``xBC`` to ``seqlen`` after ``nn.Conv1d`` but not
    after ``causal_conv1d_fn``. A length mismatch vs ``dt`` then fails
    ``assert dt.shape == (batch, seqlen, nheads)`` inside ``mamba_chunk_scan_combined``.
    """
    try:
        import mamba_ssm.modules.mamba2_simple as m2s  # type: ignore
    except Exception:
        return
    cls = m2s.Mamba2Simple
    if getattr(cls, "_nlh_m2_fwd_patch", False):
        return

    orig_forward = cls.forward

    def forward_patched(self, u, seq_idx=None):  # noqa: ANN001
        if self.use_mem_eff_path:
            return orig_forward(self, u, seq_idx=seq_idx)

        from einops import rearrange, repeat  # type: ignore

        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # type: ignore

        _, seqlen, _ = u.shape
        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log)
        initial_states = (
            repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        )
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)
        assert self.activation in ["silu", "swish"]

        if m2s.causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
            xBC = xBC[:, :seqlen, :]
        else:
            xBC = m2s.causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)
            xBC = xBC[:, :seqlen, :]

        L_bc = int(xBC.shape[1])
        if L_bc != seqlen:
            z = z[:, :L_bc, :]
            dt = dt[:, :L_bc, :]
            if seq_idx is not None:
                seq_idx = seq_idx[:, :L_bc]

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        return self.out_proj(y)

    cls.forward = forward_patched
    cls._nlh_m2_fwd_patch = True


def _mamba2_padded_d_model(dim: int, block_cls: type) -> tuple[int, int, int]:
    """
    Mamba2Simple requires (expand * d_model) % headdim == 0.
    Defaults vary by release (e.g. headdim=128 on main, older wheels used 64).
    Read defaults from the block constructor and round ``dim`` up minimally.
    """
    d0 = int(dim)
    if d0 < 1:
        raise ValueError("dim must be >= 1")
    sig = inspect.signature(block_cls.__init__)
    expand = 2
    if "expand" in sig.parameters:
        ed = sig.parameters["expand"].default
        if ed is not inspect.Parameter.empty:
            expand = int(ed)
    headdim = 128
    if "headdim" in sig.parameters:
        hd = sig.parameters["headdim"].default
        if hd is not inspect.Parameter.empty:
            headdim = int(hd)
    g = math.gcd(int(expand), int(headdim))
    step = int(headdim) // g
    inner = ((d0 + step - 1) // step) * step
    return inner, int(expand), int(headdim)


class _MambaBaseline(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        variant: Literal["mamba1", "mamba2"],
        d_state: int = 64,
    ) -> None:
        super().__init__()
        self._variant = variant
        Block = _resolve_mamba_block(variant)
        self._in_proj = nn.Identity()
        self._out_proj = nn.Identity()
        d_use = int(dim)
        d_state = max(1, int(d_state))
        # Keep defaults lightweight and comparable to other baselines.
        if variant == "mamba1":
            self.backbone = Block(d_model=d_use, d_state=min(d_state, 256), d_conv=4, expand=2)
        else:
            _force_mamba2_torch_conv1d()
            inner, m_expand, m_headdim = _mamba2_padded_d_model(dim, Block)
            if inner != dim:
                self._in_proj = nn.Linear(dim, inner, bias=False)
                self._out_proj = nn.Linear(inner, dim, bias=False)
                d_use = inner
            # Try common constructor signatures across Mamba2 releases.
            kw: dict = dict(d_model=d_use, d_state=d_state, d_conv=4, expand=m_expand)
            sig_b = inspect.signature(Block.__init__)
            if "headdim" in sig_b.parameters:
                kw["headdim"] = m_headdim
            # Fused Triton + causal_conv1d path can require stride alignment; non-fused is slower but robust.
            if "use_mem_eff_path" in sig_b.parameters:
                kw["use_mem_eff_path"] = False
            try:
                self.backbone = Block(**kw)
            except TypeError:
                kw.pop("headdim", None)
                try:
                    self.backbone = Block(**kw)
                except TypeError:
                    kw.pop("use_mem_eff_path", None)
                    try:
                        self.backbone = Block(**kw)
                    except TypeError:
                        self.backbone = Block(d_model=d_use)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, h_meta: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002
        # causal_conv1d / fused SSD paths can be picky about memory layout on CUDA.
        x = self._in_proj(x).contiguous()
        # Mamba2 Triton SSD expects seqlen aligned with chunk_size; short windows (e.g. L=128, chunk=256)
        # otherwise trip assert dt.shape == (batch, seqlen, nheads) inside mamba_chunk_scan_combined.
        pad_len = 0
        if self._variant == "mamba2":
            cs = int(getattr(self.backbone, "chunk_size", 256) or 256)
            if cs > 0:
                _, L, _ = x.shape
                pad_len = (-L) % cs
                if pad_len:
                    x = F.pad(x, (0, 0, 0, pad_len))
        y = self.backbone(x)
        if pad_len:
            y = y[:, : x.size(1) - pad_len, :]
        y = self._out_proj(y).contiguous()
        return self.proj(y)


def _infer_dim_from_batch(batch: Tuple[torch.Tensor, torch.Tensor]) -> int:
    x, _h = batch
    if x.dim() != 3:
        raise ValueError("Expected X shape (B, L, D)")
    return x.size(-1)


def _train_one_epoch(
    model: nn.Module,
    dl,
    opt: optim.Optimizer,
    device: torch.device,
    *,
    grad_clip_norm: float | None = None,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, h in dl:
        x = x.to(device).contiguous()
        h = h.to(device)
        opt.zero_grad(set_to_none=True)
        y = model(x, h)
        # Next-step prediction within window
        loss = ((y[:, :-1, :] - x[:, 1:, :]) ** 2).mean()
        loss.backward()
        if grad_clip_norm is not None and float(grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        opt.step()
        total += float(loss.detach().cpu())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def _eval(
    model: nn.Module,
    dl,
    device: torch.device,
    hierarchy: HierarchyMeta,
    *,
    smape_baseline: float | None = None,
) -> dict:
    model.eval()
    smapes = []
    rmsse_vals = []
    acd_vals = []
    tw_mse_vals = []
    crps_vals = []

    # expected hierarchical distance between timesteps: |depth_i - depth_j|
    # depth is given in H: (B, L, 1)
    for x, h in dl:
        x = x.to(device).contiguous()
        h = h.to(device)
        y = model(x, h)

        y_pred = y[:, :-1, :]
        y_true = x[:, 1:, :]

        smapes.append(smape(y_pred, y_true, reduction="mean").detach().cpu())

        # RMSSE uses the in-window history as proxy train series
        # y_train: (B, T) per-dim; flatten dims into batch for simplicity
        b, l, d = x.shape
        y_train = x[:, :-1, :].reshape(b * d, l - 1)
        rmsse_batch = rmsse(
            y_pred.reshape(b * d, l - 1),
            y_true.reshape(b * d, l - 1),
            y_train,
            seasonality=1,
            reduction="mean",
        )
        rmsse_vals.append(rmsse_batch.detach().cpu())

        tw_mse_vals.append(tw_mse(y_pred, y_true, reduction="mean").detach().cpu())

        # Proxy probabilistic forecast from deterministic prediction:
        # use residual std along horizon as homoscedastic sigma.
        resid = (y_true - y_pred)
        sigma = resid.std(dim=1, keepdim=True).expand_as(y_true).clamp_min(1e-6)
        crps_vals.append(crps_gaussian(y_true, y_pred, sigma, reduction="mean").detach().cpu())

        # ACD: compare Poincaré distances of points (use predictions as points)
        # against expected depth-distance matrix from H.
        depths = h[:, :, 0]  # (B, L)
        exp_hier = (depths.unsqueeze(-1) - depths.unsqueeze(-2)).abs()  # (B, L, L)
        # Optional scale using dataset metadata to make depths comparable
        hier_scale = max(1.0, float(hierarchy.max_depth))
        exp_hier = exp_hier / hier_scale

        # acd_paper only averages pairs with expected tree distance > eps. For one
        # series per window, depth is usually constant along time → exp_hier is all
        # zeros off-diagonal → no valid pairs → ACD is trivially 0. Fall back to a
        # chain prior on time indices (normalized |i−j|) so the metric stays informative.
        eye = torch.eye(l, device=exp_hier.device, dtype=torch.bool)
        max_hier = exp_hier.masked_fill(eye, 0.0).amax(dim=(-2, -1), keepdim=True)
        idx = torch.arange(l, device=exp_hier.device, dtype=exp_hier.dtype)
        exp_time = (idx.view(1, l, 1) - idx.view(1, 1, l)).abs() / float(max(l - 1, 1))
        exp_time = exp_time.expand(b, -1, -1)
        exp_dist = torch.where(max_hier > 1e-6, exp_hier, exp_time)

        # Use curvature baseline ~1.0 for metric; if you log per-step c later, plug it in here.
        acd_vals.append(acd(y, exp_dist, c=1.0, reduction="mean").detach().cpu())

    out = {
        "smape": float(torch.stack(smapes).mean()) if smapes else 0.0,
        "rmsse": float(torch.stack(rmsse_vals).mean()) if rmsse_vals else 0.0,
        "tw_mse": float(torch.stack(tw_mse_vals).mean()) if tw_mse_vals else 0.0,
        "crps": float(torch.stack(crps_vals).mean()) if crps_vals else 0.0,
        "acd": float(torch.stack(acd_vals).mean()) if acd_vals else 0.0,
    }
    vram = peak_vram_gb(device)
    out["peak_vram_gb"] = float(vram)
    out["mat"] = (
        float(mat_score(smape_baseline - out["smape"], vram, maximize_delta=True))
        if (smape_baseline is not None and vram > 0)
        else None
    )
    return out


def _resolve_device(device_str: str | None) -> torch.device:
    """Pick torch.device from CLI; default cuda if available else cpu."""
    if device_str is None or not str(device_str).strip():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(str(device_str).strip())
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"Requested device {dev} but CUDA is not available.")
    if dev.type == "cuda" and dev.index is not None and dev.index >= torch.cuda.device_count():
        raise SystemExit(
            f"Requested device {dev} but only {torch.cuda.device_count()} CUDA device(s) are visible."
        )
    return dev


def _dist_worker_info() -> Optional[Tuple[int, int, int]]:
    """If launched with ``torchrun`` / DDP (WORLD_SIZE>1), return (local_rank, world_size, rank)."""
    if "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        return None
    ws = int(os.environ["WORLD_SIZE"])
    if ws <= 1:
        return None
    return int(os.environ["LOCAL_RANK"]), ws, int(os.environ.get("RANK", "0"))


def _sanitize_json(obj: object) -> object:
    """Replace non-finite floats with None so ``json.dumps(..., allow_nan=False)`` succeeds."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _build_train_eval_loaders(
    dataset_path: Path,
    *,
    distributed: bool,
    rank: int,
    world_size: int,
    normalize_window: bool,
    seq_len: int,
    stride: int,
) -> Tuple[DataLoader, DataLoader, HierarchyMeta, int, int]:
    import sys

    from torch.utils.data.distributed import DistributedSampler

    seq_len_e = int(seq_len)
    stride_e = int(stride)
    try:
        mx = max_series_length_long(dataset_path, id_col="id", time_col="time")
        if mx < 2:
            raise SystemExit(
                f"Dataset {dataset_path}: longest series has only {mx} row(s); need at least 2 for seq_len."
            )
        seq_naive = min(seq_len_e, mx)
        st_naive = min(stride_e, mx)
        if st_naive < 1:
            st_naive = 1
        if seq_len_e > mx:
            print(
                f"[run_exp] seq_len={seq_len_e} exceeds longest series ({mx} rows); "
                f"capping to seq_len<={mx} (stride may be relaxed for enough windows).",
                file=sys.stderr,
            )
        seq_len_e, stride_e, _nw = resolve_seq_stride_long(
            dataset_path,
            seq_len_e,
            stride_e,
            id_col="id",
            time_col="time",
            min_windows=32,
        )
        if (seq_len_e, stride_e) != (seq_naive, st_naive):
            print(
                f"[run_exp] windowing adjusted for coverage: seq_len={seq_len_e}, stride={stride_e} "
                f"(naive cap would be seq_len={seq_naive}, stride={st_naive}; ~{_nw} windows). "
                "Override with --seq_len / --stride if needed.",
                file=sys.stderr,
            )
    except SystemExit:
        raise
    except Exception:
        # Missing columns or unreadable path; let get_dataloader report.
        pass

    batch_size = 32
    kw = dict(
        dataset="custom",
        path=dataset_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        seq_len=seq_len_e,
        stride=stride_e,
        schema="long",
        id_col="id",
        time_col="time",
        value_cols=("value",),
        normalize_window=normalize_window,
    )
    dl_train, hierarchy = get_dataloader(**kw, hierarchy=None)
    ds_train = dl_train.dataset
    if distributed:
        train_sampler = DistributedSampler(
            ds_train, num_replicas=world_size, rank=rank, shuffle=True, seed=0
        )
        dl_train = DataLoader(
            ds_train,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,
            drop_last=False,
        )
    else:
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    dl_eval, _ = get_dataloader(**kw, hierarchy=hierarchy)
    return dl_train, dl_eval, hierarchy, seq_len_e, stride_e


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        required=True,
        choices=["mamba1", "mamba2", "nlh_ssm", "transformer", "informer"],
    )
    p.add_argument("--dataset", required=True, help="Path to CSV/Parquet dataset file")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW LR for baselines (Transformer, Mamba, Informer).")
    p.add_argument(
        "--nlh_lr",
        type=float,
        default=3e-4,
        help="AdamW LR when --model nlh_ssm (hyperbolic stack; default 3e-4 is gentler than 1e-3). Ignored for other models.",
    )
    p.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Per-step gradient L2 clip for nlh_ssm (0 disables). Baselines ignore this by default.",
    )
    p.add_argument(
        "--nlh_num_layers",
        type=int,
        default=2,
        help="MixerSeqSimple depth when --model nlh_ssm.",
    )
    p.add_argument(
        "--nlh_expand",
        type=int,
        default=2,
        help="MixerSeqSimple expand factor when --model nlh_ssm.",
    )
    p.add_argument(
        "--nlh_c_base",
        type=float,
        default=0.05,
        help="Initial curvature scale for ACG when --model nlh_ssm (smaller often stabilizes training).",
    )
    p.add_argument(
        "--adamw_weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay for all models.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Single-process device: cuda:0, cuda:1, cpu. Ignored under multi-GPU DDP (use YAML + run_main).",
    )
    p.add_argument(
        "--cuda_indices",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids for DDP (passed by run_main when YAML lists multiple GPUs).",
    )
    p.add_argument(
        "--smape_baseline",
        type=float,
        default=None,
        help="Optional baseline sMAPE for MAT computation: (baseline - model) / peak_vram_gb",
    )
    p.add_argument(
        "--normalize-window",
        dest="normalize_window",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Z-score each (L, D) window over time before batching (default: on; stabilizes nlh_ssm on raw-scale CSVs).",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sliding window length (default 128). Short clinical paths need smaller values, e.g. 8–32.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride (default: same as --seq_len). Use a smaller stride for more overlapping windows.",
    )
    p.add_argument(
        "--nlh_hparams_file",
        type=str,
        default=None,
        help="YAML with nlh_ssm defaults/overrides (from scripts/tune_nlh_ssm.py). Merged by dataset stem.",
    )
    p.add_argument(
        "--nlh_ablation",
        type=str,
        default=None,
        choices=["wo_hyp", "wo_acg", "wo_ph_scan"],
        help="NL-H-H-SSM component ablation: wo_hyp (Euclidean scan), wo_acg (fixed c=1), wo_ph_scan (sequential Möbius).",
    )
    p.add_argument(
        "--force_nlh_c_base",
        type=float,
        default=None,
        help="Override nlh_c_base after YAML merge (parameter-sensitivity sweeps).",
    )
    p.add_argument(
        "--force_nlh_expand",
        type=int,
        default=None,
        help="Override nlh_expand after YAML merge (capacity sweeps).",
    )
    p.add_argument(
        "--mamba_d_state",
        type=int,
        default=64,
        help="Mamba/Mamba-2 state dimension (default 64; use in sensitivity sweeps).",
    )
    args = p.parse_args()
    if args.model == "mamba2":
        _require_triton_language_cumsum_for_mamba2()

    dataset_path = Path(args.dataset)
    dataset_name = dataset_path.stem
    if args.model == "nlh_ssm":
        _apply_nl_hparams_yaml(args, dataset_name)
        if getattr(args, "force_nlh_c_base", None) is not None:
            args.nlh_c_base = float(args.force_nlh_c_base)
        if getattr(args, "force_nlh_expand", None) is not None:
            args.nlh_expand = int(args.force_nlh_expand)

    if int(args.seq_len) < 2:
        raise SystemExit("--seq_len must be >= 2.")
    stride = int(args.stride) if args.stride is not None else int(args.seq_len)
    if stride < 1:
        raise SystemExit("--stride must be >= 1.")

    dist_info = _dist_worker_info()
    distributed = dist_info is not None
    rank = dist_info[2] if dist_info else 0

    if distributed:
        if not torch.cuda.is_available():
            raise SystemExit("DDP requires CUDA in this training script.")
        if not args.cuda_indices:
            raise SystemExit("DDP requires --cuda_indices (set by run_main from YAML).")
        phys_ids = [int(x.strip()) for x in args.cuda_indices.split(",") if x.strip()]
        if rank >= len(phys_ids):
            raise SystemExit("RANK is out of range for the given cuda_indices list.")
        phys = phys_ids[rank]
        try:
            dist.init_process_group(backend="nccl")
        except Exception:
            dist.init_process_group(backend="gloo")
        device = torch.device(f"cuda:{phys}")
        torch.cuda.set_device(device)
    else:
        device = _resolve_device(args.device)

    dl_train, dl_eval, hierarchy, seq_len_eff, stride_eff = _build_train_eval_loaders(
        dataset_path,
        distributed=distributed,
        rank=rank,
        world_size=dist_info[1] if dist_info else 1,
        normalize_window=args.normalize_window,
        seq_len=int(args.seq_len),
        stride=stride,
    )

    dim = _infer_dim_from_batch(next(iter(dl_train)))

    model_name: ModelName = args.model
    if model_name == "nlh_ssm":
        nl = max(1, int(args.nlh_num_layers))
        ex = max(1, int(args.nlh_expand))
        ablation = getattr(args, "nlh_ablation", None)
        model = MixerSeqSimple(
            dim=dim,
            num_layers=nl,
            expand=ex,
            h_meta_dim=1,
            c_base=float(args.nlh_c_base),
            ablation=ablation,
        )
    elif model_name == "transformer":
        model = _TransformerBaseline(dim=dim)
    elif model_name == "informer":
        model = _InformerBaseline(dim=dim)
    elif model_name in ("mamba1", "mamba2"):
        model = _MambaBaseline(dim=dim, variant=model_name, d_state=int(args.mamba_d_state))
    else:
        raise SystemExit(f"Unsupported model: {model_name}")

    model = model.to(device)
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    lr_eff = float(args.nlh_lr) if model_name == "nlh_ssm" else float(args.lr)
    opt = optim.AdamW(
        model.parameters(),
        lr=lr_eff,
        weight_decay=float(args.adamw_weight_decay),
    )
    clip_eff: float | None = None
    if model_name == "nlh_ssm" and float(args.grad_clip_norm) > 0:
        clip_eff = float(args.grad_clip_norm)

    history = []
    try:
        for epoch in range(int(args.epochs)):
            if distributed:
                dl_train.sampler.set_epoch(epoch)
            train_loss = _train_one_epoch(model, dl_train, opt, device, grad_clip_norm=clip_eff)
            if rank == 0:
                metrics = _eval(model, dl_eval, device, hierarchy, smape_baseline=args.smape_baseline)
                history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})
            if distributed:
                dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()

    if rank != 0:
        return

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = model_name
    if model_name == "nlh_ssm" and getattr(args, "nlh_ablation", None):
        model_tag = f"{model_name}_{args.nlh_ablation}"
    out_path = out_dir / f"{dataset_name}_{model_tag}.json"

    payload = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "model": model_tag,
        "base_model": model_name,
        "nlh_ablation": getattr(args, "nlh_ablation", None),
        "device": str(device),
        "distributed": distributed,
        "cuda_indices": args.cuda_indices if distributed else None,
        "hmeta": asdict(hierarchy),
        "hyperparams": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "nlh_lr": float(args.nlh_lr) if model_name == "nlh_ssm" else None,
            "effective_optimizer_lr": lr_eff,
            "grad_clip_norm": clip_eff if model_name == "nlh_ssm" else None,
            "nlh_num_layers": int(args.nlh_num_layers) if model_name == "nlh_ssm" else None,
            "nlh_expand": int(args.nlh_expand) if model_name == "nlh_ssm" else None,
            "nlh_c_base": float(args.nlh_c_base) if model_name == "nlh_ssm" else None,
            "nlh_ablation": getattr(args, "nlh_ablation", None) if model_name == "nlh_ssm" else None,
            "adamw_weight_decay": float(args.adamw_weight_decay),
            "nlh_hparams_file": str(args.nlh_hparams_file) if getattr(args, "nlh_hparams_file", None) else None,
            "normalize_window": bool(args.normalize_window),
            "seq_len": int(seq_len_eff),
            "stride": int(stride_eff),
        },
        "history": history,
        "final": history[-1] if history else {},
    }
    out_path.write_text(
        json.dumps(_sanitize_json(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()

