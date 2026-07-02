"""Print per-dataset ranks and win counts from ``results/*_{model}.json`` (Table 6 metrics)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.generate_table_6 import TABLE6_DATASETS  # noqa: E402

DEFAULT_MODELS = ["transformer", "informer", "mamba1", "mamba2", "nlh_ssm"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank Table-6 models from results JSON files.")
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    args = ap.parse_args()

    rd = Path(args.results_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    wins = {m: 0 for m in models}
    nlh_top1 = 0
    nlh_top2 = 0

    print(f"Results: {rd.resolve()}\n")
    for spec in TABLE6_DATASETS:
        vals: dict[str, float] = {}
        windows: dict[str, tuple[int, int] | None] = {}
        for m in models:
            p = rd / f"{spec.key}_{m}.json"
            if not p.is_file():
                continue
            payload = json.loads(p.read_text(encoding="utf-8"))
            fin = payload.get("final") or {}
            if spec.metric in fin:
                vals[m] = float(fin[spec.metric])
            hp = payload.get("hyperparams") or {}
            sl, st = hp.get("seq_len"), hp.get("stride")
            if sl is not None and st is not None:
                windows[m] = (int(sl), int(st))
            else:
                windows[m] = None

        if not vals:
            print(f"{spec.key:12}  (no json)\n")
            continue

        ranked = sorted(vals.items(), key=lambda x: x[1])
        best_m, best_v = ranked[0]
        wins[best_m] = wins.get(best_m, 0) + 1
        nlh_rank = next((i + 1 for i, (m, _) in enumerate(ranked) if m == "nlh_ssm"), None)
        if nlh_rank == 1:
            nlh_top1 += 1
        if nlh_rank is not None and nlh_rank <= 2:
            nlh_top2 += 1

        nlh_v = vals.get("nlh_ssm")
        nlh_w = windows.get("nlh_ssm")
        wnote = f" nlh_win={nlh_w}" if nlh_w else ""
        print(
            f"{spec.key:12} {spec.metric:6}  #1 {best_m:12} {best_v:.4f}"
            + (f"  nlh #{nlh_rank} {nlh_v:.4f}{wnote}" if nlh_v is not None else "")
        )
        print("             " + "  ".join(f"{m}:{v:.4f}" for m, v in ranked))
        # flag mixed window protocols
        uniq = {windows[m] for m in ranked if windows.get(m)}
        if len(uniq) > 1:
            print("             [!] mixed seq_len/stride across models:", windows)
        print()

    print("Wins (lower is better):")
    for m in sorted(wins, key=lambda k: -wins.get(k, 0)):
        print(f"  {m:12} {wins.get(m, 0)}/10")
    print(f"\nnlh_ssm: {nlh_top1}/10 first, {nlh_top2}/10 top-2")


if __name__ == "__main__":
    main()
