#!/usr/bin/env python3
"""Aggregate benchmarks/sensitivity_results/*.json -> figure10 JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def _acd(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    v = (data.get("final") or {}).get("acd")
    if v is None:
        raise KeyError(f"acd missing in {path}")
    return float(v)


def _normalize_lower_better(values: List[float]) -> List[float]:
    if not values:
        return values
    best = min(values)
    worst = max(values)
    span = worst - best
    if span < 1e-12:
        return [1.0 for _ in values]
    return [(worst - v) / span for v in values]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="benchmarks/sensitivity_results")
    ap.add_argument("--out", type=str, default="benchmarks/figure10_parameter_sensitivity.json")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    c_vals = [0.1, 0.5, 1.0, 1.5, 2.0]
    acd_vals = [_acd(in_dir / f"logic_cbase_{c}.json") for c in c_vals]

    d_state = [16, 32, 64, 128]
    nlh_raw = [_acd(in_dir / f"logic_expand_{ds // 16}.json") for ds in d_state]
    mamba_raw = [_acd(in_dir / f"logic_mamba2_dstate_{ds}.json") for ds in d_state]
    nlh_norm = _normalize_lower_better(nlh_raw)
    mamba_norm = _normalize_lower_better(mamba_raw)

    payload = {
        "mode": "measured",
        "note": "Logic dataset; left=raw ACD (lower better); right=normalized ACD (1=best in sweep)",
        "c": c_vals,
        "acd": acd_vals,
        "d_state": d_state,
        "nlh_norm": nlh_norm,
        "mamba_norm": mamba_norm,
        "nlh_acd_raw": nlh_raw,
        "mamba_acd_raw": mamba_raw,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print("  c vs ACD:", list(zip(c_vals, acd_vals)))
    print("  d_state nlh norm:", list(zip(d_state, nlh_norm)))
    print("  d_state mamba norm:", list(zip(d_state, mamba_norm)))


if __name__ == "__main__":
    main()
