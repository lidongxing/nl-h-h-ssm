"""
Unified dataloader for NL-H-H-SSM.

Goal:
- Load time-series datasets from CSV/Parquet (M5, Wiki-Traffic, Expert-Log, ...).
- Extract hierarchy metadata required by ACG (max_depth, avg_branching_factor).
- Provide get_dataloader() that yields (X, H) pairs, where H is the hierarchical
  level/depth signal for the sequence (broadcastable to (B, L, Hm)).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

DatasetName = Literal[
    "m5",
    "wiki-traffic",
    "expert-log",
    "custom",
]


@dataclass(frozen=True)
class HierarchyMeta:
    name: str
    max_depth: int
    avg_branching_factor: float
    # Optional delimiter used to infer depth from item IDs like "a/b/c"
    id_delim: str = "/"


# Default hierarchy priors (can be overridden per call).
HIERARCHY_PRESETS: Dict[str, HierarchyMeta] = {
    "m5": HierarchyMeta(name="m5", max_depth=5, avg_branching_factor=3.0, id_delim="/"),
    "wiki-traffic": HierarchyMeta(name="wiki-traffic", max_depth=4, avg_branching_factor=4.5, id_delim="/"),
    "expert-log": HierarchyMeta(name="expert-log", max_depth=20, avg_branching_factor=2.0, id_delim="/"),
}


def _read_table(path: Union[str, Path]):
    """
    Read CSV/Parquet into a pandas DataFrame-like object.

    We avoid forcing heavyweight deps in setup.py; user environments differ.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Reading CSV/Parquet requires pandas. Install with `pip install pandas pyarrow`."
        ) from e

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {suffix}")


def max_series_length_long(
    path: Union[str, Path],
    *,
    id_col: str = "id",
    time_col: str = "time",
) -> int:
    """
    Longest run length (row count per ``id``) for a long-format table.

    Reads only ``id_col`` and ``time_col`` for speed on large CSVs.
    """
    path = Path(path)
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Reading CSV/Parquet requires pandas. Install with `pip install pandas pyarrow`."
        ) from e

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, usecols=[id_col, time_col])
    elif suffix in {".parquet"}:
        df = pd.read_parquet(path, columns=[id_col, time_col])
    else:
        df = _read_table(path)
        if id_col not in df.columns or time_col not in df.columns:
            raise ValueError(
                f"Expected columns {id_col!r}, {time_col!r} in {path}; got {list(df.columns)}"
            )
        df = df[[id_col, time_col]]
    if id_col not in df.columns or time_col not in df.columns:
        raise ValueError(
            f"Expected columns {id_col!r}, {time_col!r} in {path}; got {list(df.columns)}"
        )
    return int(df.groupby(id_col).size().max())


def series_lengths_long(
    path: Union[str, Path],
    *,
    id_col: str = "id",
    time_col: str = "time",
) -> List[int]:
    """Row count per series id (same ordering as groupby iteration in sequence build)."""
    path = Path(path)
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Reading CSV/Parquet requires pandas. Install with `pip install pandas pyarrow`."
        ) from e

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, usecols=[id_col, time_col])
    elif suffix in {".parquet"}:
        df = pd.read_parquet(path, columns=[id_col, time_col])
    else:
        df = _read_table(path)
        df = df[[id_col, time_col]]
    return [int(x) for x in df.groupby(id_col, sort=False).size().tolist()]


def sliding_window_count_long(lengths: Sequence[int], seq_len: int, stride: int) -> int:
    """
    Number of sliding windows that ``_build_sequences_from_long_df`` would emit.

    Matches ``for start in range(0, L - seq_len + 1, stride)``.
    """
    if seq_len < 1 or stride < 1:
        return 0
    total = 0
    for L in lengths:
        if L < seq_len:
            continue
        total += (L - seq_len) // stride + 1
    return int(total)


def resolve_seq_stride_long(
    path: Union[str, Path],
    seq_len_req: int,
    stride_req: int,
    *,
    id_col: str = "id",
    time_col: str = "time",
    min_windows: int = 32,
    min_seq_len: int = 4,
) -> Tuple[int, int, int]:
    """
    Effective ``(seq_len, stride)`` for long-format CSVs when the naive cap
    (``seq_len = min(request, max_series_length)`` with large stride) would
    leave almost no windows (e.g. medical: one series reaches max length).

    Prefer ``stride=1`` then shrink ``seq_len`` (down to ``min_seq_len``) until
    the sliding-window count reaches ``target``, where
    ``target = max(2, min(min_windows, max(n_naive, 1)))`` and ``n_naive`` is the
    count under the initial cap. That way sparse but valid settings (e.g. four
    non-overlapping tourism blocks at 128/128) are preserved, while datasets
    with fewer than two windows are expanded.
    """
    lengths = series_lengths_long(path, id_col=id_col, time_col=time_col)
    if not lengths:
        return max(1, int(seq_len_req)), max(1, int(stride_req)), 0
    mx = max(lengths)
    capped = min(int(seq_len_req), mx)
    seq_len_e = int(capped)
    stride_e = min(int(stride_req), mx)
    if stride_e < 1:
        stride_e = 1
    lo_bound = max(min(2, mx), min(int(min_seq_len), mx))

    def nwin(sl: int, st: int) -> int:
        return sliding_window_count_long(lengths, sl, st)

    nw = nwin(seq_len_e, stride_e)
    # Do not force ``min_windows`` when the naive cap already yields few but usable
    # windows (e.g. tourism ~4 blocks at seq_len=stride=128); only relax when we
    # are short of ``min_windows`` *or* below this adaptive floor.
    target = max(2, min(int(min_windows), max(nw, 1)))
    if nw >= target:
        return seq_len_e, stride_e, nw

    if stride_e > 1:
        stride_e = 1
        nw = nwin(seq_len_e, stride_e)
    if nw >= target:
        return seq_len_e, stride_e, nw

    lo, hi = int(lo_bound), int(seq_len_e)
    if nwin(lo, 1) < target:
        seq_len_e = lo
        return seq_len_e, 1, nwin(seq_len_e, 1)

    while lo < hi:
        mid = (lo + hi) // 2
        if nwin(mid, 1) >= target:
            hi = mid
        else:
            lo = mid + 1
    seq_len_e = lo
    return seq_len_e, 1, nwin(seq_len_e, 1)


def _infer_depth_from_id(item_id: str, delim: str) -> int:
    # depth == number of segments - 1, so root-only "x" -> 0
    if item_id is None:
        return 0
    s = str(item_id)
    if not s:
        return 0
    return max(0, len([p for p in s.split(delim) if p != ""]) - 1)


class _SeqDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Yields:
    - X: (L, D) float tensor
    - H: (L, 1) float tensor (hierarchy depth signal, broadcastable)
    """

    def __init__(
        self,
        sequences: Sequence[torch.Tensor],
        depths: Sequence[float],
        *,
        normalize_window: bool = False,
    ) -> None:
        if len(sequences) != len(depths):
            raise ValueError("sequences and depths must have same length")
        self._seqs = sequences
        self._depths = depths
        self._normalize_window = normalize_window

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._seqs[idx]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if self._normalize_window:
            # Per channel over time (L, D): stabilizes hyperbolic / scan blocks on raw-scale CSVs.
            xm = x.mean(dim=0, keepdim=True)
            xs = x.std(dim=0, keepdim=True)
            x = torch.where(xs > 1e-6, (x - xm) / xs, x - xm)
        l = x.size(0)
        h = x.new_full((l, 1), float(self._depths[idx]))
        return x, h


def _build_sequences_from_long_df(
    df,
    *,
    id_col: str,
    time_col: str,
    value_cols: Sequence[str],
    seq_len: int,
    stride: int,
    depth_fn: Callable[[str], int],
) -> Tuple[List[torch.Tensor], List[float]]:
    # group by series id, sort by time, then sliding windows
    sequences: List[torch.Tensor] = []
    depths: List[float] = []

    # Pandas-like API (works with pandas df)
    for sid, g in df.groupby(id_col):
        g2 = g.sort_values(time_col)
        vals = torch.tensor(g2[list(value_cols)].to_numpy(), dtype=torch.float32)
        if vals.size(0) < seq_len:
            continue
        d = float(depth_fn(str(sid)))
        for start in range(0, vals.size(0) - seq_len + 1, stride):
            sequences.append(vals[start : start + seq_len])
            depths.append(d)
    return sequences, depths


def _build_sequences_from_wide_df(
    df,
    *,
    id_col: str,
    feature_cols: Sequence[str],
    seq_len: int,
    stride: int,
    depth_fn: Callable[[str], int],
) -> Tuple[List[torch.Tensor], List[float]]:
    # Each row is one series; feature_cols are timesteps (and/or multivariate features).
    sequences: List[torch.Tensor] = []
    depths: List[float] = []

    ids = df[id_col].astype(str).tolist()
    mat = torch.tensor(df[list(feature_cols)].to_numpy(), dtype=torch.float32)  # (N, T)
    for i, sid in enumerate(ids):
        v = mat[i]
        if v.numel() < seq_len:
            continue
        d = float(depth_fn(sid))
        for start in range(0, v.numel() - seq_len + 1, stride):
            sequences.append(v[start : start + seq_len].unsqueeze(-1))
            depths.append(d)
    return sequences, depths


def get_dataloader(
    dataset: DatasetName,
    path: Union[str, Path],
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    seq_len: int = 128,
    stride: int = 128,
    # Schema options
    schema: Literal["long", "wide"] = "long",
    id_col: str = "id",
    time_col: str = "time",
    value_cols: Optional[Sequence[str]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    # Hierarchy options
    hierarchy: Optional[HierarchyMeta] = None,
    id_delim: Optional[str] = None,
    # Optional explicit depth column (overrides inference)
    depth_col: Optional[str] = None,
    # Z-score each (L, D) window over time (recommended for nlh_ssm on arbitrary CSV scales).
    normalize_window: bool = False,
) -> Tuple[DataLoader, HierarchyMeta]:
    """
    Build a unified DataLoader that yields (X, H) pairs.

    Returns:
    - dataloader: batches of (X, H) with shapes (B, L, D) and (B, L, 1)
    - hierarchy_meta: (max_depth, avg_branching_factor) used for the dataset
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    if hierarchy is None:
        if dataset != "custom":
            hierarchy = HIERARCHY_PRESETS[dataset]
        else:
            hierarchy = HierarchyMeta(name="custom", max_depth=1, avg_branching_factor=1.0)

    delim = id_delim or hierarchy.id_delim

    df = _read_table(path)

    if depth_col is not None and depth_col in df.columns:
        def depth_fn(sid: str) -> int:  # type: ignore[no-redef]
            # unused when explicit depth_col present for long format;
            # for wide format we'll still infer from id.
            return _infer_depth_from_id(sid, delim)
    else:
        def depth_fn(sid: str) -> int:  # type: ignore[no-redef]
            return _infer_depth_from_id(sid, delim)

    if schema == "long":
        if value_cols is None:
            value_cols = ("value",)
        for col in (id_col, time_col, *value_cols):
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' for schema='long'")

        if depth_col is not None and depth_col in df.columns:
            # Use depth per-row but keep it consistent per series (take max).
            depth_map = df.groupby(id_col)[depth_col].max().to_dict()

            def depth_fn2(sid: str) -> int:
                return int(depth_map.get(sid, 0))

            depth_used = depth_fn2
        else:
            depth_used = depth_fn

        seqs, depths = _build_sequences_from_long_df(
            df,
            id_col=id_col,
            time_col=time_col,
            value_cols=value_cols,
            seq_len=seq_len,
            stride=stride,
            depth_fn=depth_used,
        )
    elif schema == "wide":
        if feature_cols is None:
            # Default: use all columns except id_col
            feature_cols = [c for c in df.columns if c != id_col]
        if id_col not in df.columns:
            raise ValueError(f"Missing column '{id_col}' for schema='wide'")
        if len(feature_cols) == 0:
            raise ValueError("feature_cols is empty")
        seqs, depths = _build_sequences_from_wide_df(
            df,
            id_col=id_col,
            feature_cols=feature_cols,
            seq_len=seq_len,
            stride=stride,
            depth_fn=depth_fn,
        )
    else:
        raise ValueError("schema must be 'long' or 'wide'")

    if len(seqs) == 0:
        hint = ""
        if schema == "long" and id_col in df.columns:
            mx = int(df.groupby(id_col).size().max())
            hint = (
                f" Longest series has {mx} row(s), but seq_len={seq_len}. "
                f"Use a smaller --seq_len (≤{mx}) and optionally --stride (e.g. half of seq_len) "
                "so sliding windows fit."
            )
        elif schema == "wide" and feature_cols:
            hint = (
                f" Each row has {len(feature_cols)} timestep column(s), but seq_len={seq_len}. "
                "Lower --seq_len or use a long-format CSV (id, time, value)."
            )
        raise RuntimeError(
            "No sequences were built (every series shorter than seq_len, or wrong columns). "
            "Check id/time/value_cols for schema='long', or id/feature_cols for schema='wide'."
            + hint
        )

    ds = _SeqDataset(seqs, depths, normalize_window=normalize_window)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    return dl, hierarchy

