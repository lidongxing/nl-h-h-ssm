"""
Parse ``device`` from experiment YAML (or CLI) for single-GPU vs multi-GPU DDP.

Conventions
-----------
- **CPU**: ``device: cpu``
- **单卡 GPU**: ``device: cuda:0`` 或 ``device: "cuda:1"``（与 PyTorch 字符串一致）
- **多卡并行 (DDP)**: ``device: "cuda:0,1"`` 或 ``device: "cuda:0，1"``（支持中文逗号），
  或 YAML 列表 ``device: [0, 1]``（表示物理 GPU 0 与 1）
- **自动**: 省略 ``device`` 或空，则单进程 ``cuda``（有则默认 0 号）否则 ``cpu``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass(frozen=True)
class DeviceLaunch:
    """How ``run_main`` should spawn ``run_exp``."""

    mode: str  # "single" | "ddp"
    device_arg: Optional[str]  # pass to ``run_exp --device`` when mode == single
    cuda_indices: Optional[Tuple[int, ...]]  # physical GPU indices when mode == ddp


def _split_cuda_indices(s: str) -> List[int]:
    """Parse ``0,1,2`` or ``0，1`` into ints; allow trailing junk like ``...``."""
    parts = re.split(r"[,，]", s)
    out: List[int] = []
    for p in parts:
        t = p.strip()
        if not t or t == "...":
            continue
        if not t.isdigit():
            raise ValueError(f"Invalid GPU index segment in device list: {p!r}")
        out.append(int(t))
    return out


def parse_device_field(raw: Any) -> DeviceLaunch:
    if raw is None:
        return DeviceLaunch(mode="single", device_arg=None, cuda_indices=None)
    if isinstance(raw, (list, tuple)):
        ids = [int(x) for x in raw]
        if not ids:
            return DeviceLaunch(mode="single", device_arg=None, cuda_indices=None)
        if len(ids) == 1:
            return DeviceLaunch(mode="single", device_arg=f"cuda:{ids[0]}", cuda_indices=None)
        return DeviceLaunch(mode="ddp", device_arg=None, cuda_indices=tuple(ids))

    s = str(raw).strip()
    if not s:
        return DeviceLaunch(mode="single", device_arg=None, cuda_indices=None)

    low = s.lower()
    if low == "cpu":
        return DeviceLaunch(mode="single", device_arg="cpu", cuda_indices=None)

    # Multi: "cuda:0,1" or "cuda:0，1，2" or "0,1" (shorthand for cuda GPUs)
    s_norm = s.replace("，", ",").replace(" ", "")
    if s_norm.lower().startswith("cuda:"):
        rest = s_norm.split(":", 1)[1]
        if "," in rest:
            ids = _split_cuda_indices(rest)
            if len(ids) < 2:
                raise ValueError(f"Multi-GPU device list needs at least two indices: {raw!r}")
            return DeviceLaunch(mode="ddp", device_arg=None, cuda_indices=tuple(ids))
        # cuda:N single
        if rest.isdigit():
            return DeviceLaunch(mode="single", device_arg=f"cuda:{int(rest)}", cuda_indices=None)
        raise ValueError(f"Unrecognized cuda device string: {raw!r}")

    # Shorthand "0,1" -> cuda DDP; bare "0" -> cuda:0
    if s_norm.isdigit():
        return DeviceLaunch(mode="single", device_arg=f"cuda:{int(s_norm)}", cuda_indices=None)
    if "," in s_norm and all(p.strip().isdigit() for p in s_norm.split(",") if p.strip()):
        ids = [int(p.strip()) for p in s_norm.split(",") if p.strip().isdigit()]
        if len(ids) >= 2:
            return DeviceLaunch(mode="ddp", device_arg=None, cuda_indices=tuple(ids))
        if len(ids) == 1:
            return DeviceLaunch(mode="single", device_arg=f"cuda:{ids[0]}", cuda_indices=None)

    # Single token: cuda, cuda:0, or other torch.device string
    return DeviceLaunch(mode="single", device_arg=s, cuda_indices=None)
