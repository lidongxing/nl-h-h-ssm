"""
Per-dataset hyperparameter search for ``nlh_ssm`` (MixerSeqSimple).

Holds out a random subset of sliding-window indices as validation, runs a small
grid over ``nlh_lr``, ``grad_clip_norm``, ``nlh_c_base``, ``nlh_num_layers``,
picks the combo that minimizes the **Table~6 primary metric** for that dataset
(sMAPE / RMSSE / ACD), then writes ``configs/nlh_tuned.yaml`` for use with::

    python experiments/run_exp.py --model nlh_ssm --dataset data/processed/DATA.csv \\
        --nlh_hparams_file configs/nlh_tuned.yaml --epochs 20

This does **not** guarantee SOTA on every column; it is a practical starting point
for stabilizing / improving nlh_ssm relative to hand-tuned defaults.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from nlh_ssm.models.mixer_seq_simple import MixerSeqSimple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_run_exp():
    p = _ROOT / "experiments" / "run_exp.py"
    spec = importlib.util.spec_from_file_location("run_exp_dyn", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# Same keys as scripts/generate_table_6.py TABLE6_DATASETS metric column
DS_METRIC: Dict[str, str] = {
    "tourism": "smape",
    "labor": "smape",
    "prison": "smape",
    "m5": "rmsse",
    "wiki": "rmsse",
    "electricity": "rmsse",
    "traffic": "rmsse",
    "solar": "rmsse",
    "logic": "acd",
    "medical": "acd",
}

TABLE6_STEMS = list(DS_METRIC.keys())


def _window_candidates(
    csv_path: Path,
    seq_len_req: int,
    stride_req: int,
    *,
    min_windows: int = 32,
    max_windows_tune: int = 200_000,
) -> List[Tuple[int, int, int]]:
    """
    Candidate ``(seq_len, stride, n_windows)`` for tuning.

    Datasets like tourism have only a handful of non-overlapping windows at
    ``seq_len=stride=128``; searching smaller ``seq_len`` / ``stride`` is often
    more important than tweaking ``nlh_lr`` alone.
    """
    from nlh_ssm.data.loader import max_series_length_long, series_lengths_long, sliding_window_count_long

    mx = max_series_length_long(csv_path, id_col="id", time_col="time")
    st0 = int(stride_req) if stride_req else int(seq_len_req)
    raw = [
        (int(seq_len_req), st0),
        (min(128, mx), min(128, mx)),
        (min(128, mx), 1),
        (min(96, mx), min(32, mx)),
        (min(64, mx), min(16, mx)),
        (min(64, mx), 1),
        (min(32, mx), min(8, mx)),
        (min(32, mx), 1),
        (min(16, mx), 1),
        (min(8, mx), 1),
        (min(4, mx), 1),
    ]
    lengths = series_lengths_long(csv_path, id_col="id", time_col="time")
    seen: set[Tuple[int, int]] = set()
    out: List[Tuple[int, int, int]] = []
    for sl, st in raw:
        sl = max(2, min(int(sl), mx))
        st = max(1, min(int(st), mx))
        if (sl, st) in seen:
            continue
        seen.add((sl, st))
        nw = sliding_window_count_long(lengths, sl, st)
        if nw >= 2:
            out.append((sl, st, nw))
    out.sort(key=lambda t: (-t[2], t[0], t[1]))
    rich = [t for t in out if t[2] >= int(min_windows)]
    pool = rich if rich else out
    # Skip million-window settings (m5/wiki/traffic); tuning uses max_windows subsample anyway.
    bounded = [t for t in pool if t[2] <= int(max_windows_tune)]
    if bounded:
        bounded.sort(key=lambda t: (-t[2], t[0], t[1]))
        return bounded
    pool.sort(key=lambda t: (t[2], t[0], t[1]))
    return pool[:6]


def _grid(quick: bool) -> List[Dict[str, Any]]:
    lrs = [5e-5, 1.5e-4, 4e-4] if not quick else [1e-4, 3e-4]
    clips = [0.5, 1.0] if not quick else [1.0]
    cbs = [0.03, 0.07, 0.12] if not quick else [0.05, 0.1]
    layers = [1, 2]
    expands = [2]
    combos = []
    for lr, clip, cb, nl, ex in itertools.product(lrs, clips, cbs, layers, expands):
        combos.append(
            dict(nlh_lr=float(lr), grad_clip_norm=float(clip), nlh_c_base=float(cb), nlh_num_layers=int(nl), nlh_expand=int(ex))
        )
    if quick and len(combos) > 12:
        rng = random.Random(0)
        combos = rng.sample(combos, 12)
    return combos


def _tune_one(
    re,
    csv_path: Path,
    *,
    device: torch.device,
    seq_len: int,
    stride: int,
    normalize_window: bool,
    tune_epochs: int,
    max_windows: int,
    val_frac: float,
    batch_size: int,
    seed: int,
    quick: bool,
    search_windows: bool,
) -> Tuple[str, Dict[str, Any], float, int, int]:
    from nlh_ssm.data.loader import get_dataloader, max_series_length_long, resolve_seq_stride_long

    stem = csv_path.stem
    if stem not in DS_METRIC:
        raise SystemExit(f"Unknown dataset stem {stem!r}; expected one of {TABLE6_STEMS}")
    metric = DS_METRIC[stem]

    seq_len_e = int(seq_len)
    stride_e = int(stride)
    window_triples: List[Tuple[int, int, int]] = []
    if search_windows:
        window_triples = _window_candidates(csv_path, seq_len_e, stride_e)
        print(
            f"[tune_nlh_ssm] {csv_path.name}: --search_windows candidates: "
            + ", ".join(f"({sl},{st})~{nw}" for sl, st, nw in window_triples[:6])
            + (" ..." if len(window_triples) > 6 else ""),
            flush=True,
        )
    else:
        try:
            mx = max_series_length_long(csv_path, id_col="id", time_col="time")
            if mx < 2:
                raise SystemExit(f"{csv_path}: longest series has only {mx} row(s); need at least 2.")
            seq_naive = min(seq_len_e, mx)
            st_naive = min(stride_e, mx)
            if st_naive < 1:
                st_naive = 1
            if seq_len_e > mx:
                print(
                    f"[tune_nlh_ssm] {csv_path.name}: seq_len={seq_len_e} > longest series ({mx}); "
                    f"capping seq_len to {mx} (same as run_exp); stride may relax for enough windows.",
                    flush=True,
                )
            seq_len_e, stride_e, n_est = resolve_seq_stride_long(
                csv_path,
                seq_len_e,
                stride_e,
                id_col="id",
                time_col="time",
                min_windows=32,
            )
            if (seq_len_e, stride_e) != (seq_naive, st_naive):
                print(
                    f"[tune_nlh_ssm] {csv_path.name}: windowing adjusted for coverage: "
                    f"seq_len={seq_len_e}, stride={stride_e} (~{n_est} windows; naive cap "
                    f"seq_len={seq_naive}, stride={st_naive}).",
                    flush=True,
                )
            window_triples = [(seq_len_e, stride_e, n_est)]
        except SystemExit:
            raise
        except Exception:
            window_triples = [(seq_len_e, stride_e, 0)]

    grid = _grid(quick)
    best_cfg: Dict[str, Any] | None = None
    best_score = math.inf
    best_sl, best_st = seq_len_e, stride_e

    for sl, st, n_est in window_triples:
        if search_windows:
            print(
                f"[tune_nlh_ssm] {csv_path.name}: try windows seq_len={sl} stride={st} (~{n_est} total)",
                flush=True,
            )
        kw = dict(
            dataset="custom",
            path=csv_path,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            seq_len=int(sl),
            stride=int(st),
            schema="long",
            id_col="id",
            time_col="time",
            value_cols=("value",),
            normalize_window=normalize_window,
        )
        dl0, hierarchy = get_dataloader(**kw, hierarchy=None)
        ds = dl0.dataset
        n_all = len(ds)
        if n_all < 2:
            if search_windows:
                continue
            raise SystemExit(f"{csv_path}: too few windows ({n_all}) for train/val split.")

        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(n_all, generator=g)
        if max_windows > 0 and max_windows < n_all:
            perm = perm[: int(max_windows)]
        n = int(perm.numel())
        n_val = max(1, int(n * float(val_frac)))
        train_idx = perm[n_val:].tolist()
        val_idx = perm[:n_val].tolist()

        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        dim = int(ds[0][0].shape[-1])

        for cfg in grid:
            model = MixerSeqSimple(
                dim=dim,
                num_layers=max(1, int(cfg["nlh_num_layers"])),
                expand=max(1, int(cfg["nlh_expand"])),
                h_meta_dim=1,
                c_base=float(cfg["nlh_c_base"]),
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["nlh_lr"]), weight_decay=0.01)
            clip = float(cfg["grad_clip_norm"]) if float(cfg["grad_clip_norm"]) > 0 else None
            for _ in range(int(tune_epochs)):
                re._train_one_epoch(model, train_dl, opt, device, grad_clip_norm=clip)
            with torch.no_grad():
                metrics = re._eval(model, val_dl, device, hierarchy, smape_baseline=None)
            score = metrics.get(metric)
            if score is None or not math.isfinite(float(score)):
                continue
            score = float(score)
            if score < best_score:
                best_score = score
                best_cfg = dict(cfg)
                best_sl, best_st = int(sl), int(st)

    if best_cfg is None:
        best_cfg = dict(nlh_lr=3e-4, grad_clip_norm=1.0, nlh_c_base=0.05, nlh_num_layers=2, nlh_expand=2)
        best_score = float("nan")
    return stem, best_cfg, best_score, best_sl, best_st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing CSVs")
    ap.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated stems (e.g. tourism,m5,solar). Default: all Table~6 stems present on disk.",
    )
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--normalize_window", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tune_epochs", type=int, default=5)
    ap.add_argument("--max_windows", type=int, default=6000, help="Cap windows for speed (0 = use all).")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="Smaller grid + subsample combos")
    ap.add_argument("--out", type=str, default="configs/nlh_tuned.yaml")
    ap.add_argument(
        "--merge_existing",
        action="store_true",
        help="If --out already exists, keep its defaults/overrides and only update stems tuned in this run.",
    )
    ap.add_argument(
        "--search_windows",
        action="store_true",
        help="Also search seq_len/stride (important for tourism etc. with few windows at 128/128).",
    )
    args = ap.parse_args()

    re = _load_run_exp()
    device = re._resolve_device(args.device)
    stride = int(args.stride) if args.stride is not None else int(args.seq_len)
    data_dir = Path(args.data_dir)

    if args.datasets:
        stems = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        stems = [s for s in TABLE6_STEMS if (data_dir / f"{s}.csv").is_file()]

    if not stems:
        raise SystemExit("No datasets to tune (check --data_dir and CSV names).")

    overrides: Dict[str, Dict[str, Any]] = {}
    print(f"[tune_nlh_ssm] device={device}, tune_epochs={args.tune_epochs}, stems={stems}", flush=True)
    for stem in stems:
        p = data_dir / f"{stem}.csv"
        if not p.is_file():
            print(f"[skip] missing {p}", flush=True)
            continue
        print(f"\n=== tuning {stem} ===", flush=True)
        _, cfg, sc, sl, st = _tune_one(
            re,
            p,
            device=device,
            seq_len=int(args.seq_len),
            stride=stride,
            normalize_window=bool(args.normalize_window),
            tune_epochs=int(args.tune_epochs),
            max_windows=int(args.max_windows),
            val_frac=float(args.val_frac),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            quick=bool(args.quick),
            search_windows=bool(args.search_windows),
        )
        overrides[stem] = {**cfg, "seq_len": int(sl), "stride": int(st)}
        print(f"best val {DS_METRIC[stem]}={sc:.6f} -> {cfg} | windows seq_len={sl} stride={st}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    defaults: Dict[str, Any] = {
        "nlh_lr": 3e-4,
        "grad_clip_norm": 1.0,
        "nlh_c_base": 0.05,
        "nlh_num_layers": 2,
        "nlh_expand": 2,
        "adamw_weight_decay": 0.01,
    }
    overrides_out: Dict[str, Dict[str, Any]] = dict(overrides)
    if bool(args.merge_existing) and out_path.is_file():
        try:
            old = yaml.safe_load(out_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"[tune_nlh_ssm] merge_existing: could not read {out_path}: {e}", flush=True)
        else:
            if isinstance(old.get("defaults"), dict):
                defaults = {**defaults, **old["defaults"]}
            prev = old.get("overrides")
            if isinstance(prev, dict):
                n_prev = len(prev)
                overrides_out = {**prev, **overrides}
                print(
                    f"[tune_nlh_ssm] merged {n_prev} existing override stem(s) from {out_path.name}; "
                    f"updated {list(overrides.keys())}.",
                    flush=True,
                )
    payload = {"defaults": defaults, "overrides": overrides_out}
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\nWrote {out_path.resolve()}", flush=True)
    print(
        "Full training example:\n"
        f"  python experiments/run_exp.py --model nlh_ssm --dataset {data_dir}/tourism.csv "
        f"--nlh_hparams_file {out_path} --epochs 20",
        flush=True,
    )


if __name__ == "__main__":
    main()
