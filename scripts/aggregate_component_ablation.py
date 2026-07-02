#!/usr/bin/env python3
"""Build benchmarks/ablation_model_components.json from results/* ablation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


KEYS = ["full", "wo_hyp", "wo_acg", "wo_ph_scan"]
FILE_SUFFIX = {
    "full": "nlh_ssm",
    "wo_hyp": "nlh_ssm_wo_hyp",
    "wo_acg": "nlh_ssm_wo_acg",
    "wo_ph_scan": "nlh_ssm_wo_ph_scan",
}


def _read_final(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    final = data.get("final") or {}
    if not isinstance(final, dict):
        raise ValueError(f"No final metrics in {path}")
    return final


def _metric(final: Dict[str, Any], name: str) -> float:
    v = final.get(name)
    if v is None or not isinstance(v, (int, float)):
        raise KeyError(f"Missing metric {name!r} in {final}")
    return float(v)


def _load_series(results_dir: Path, stem: str, metric: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, suffix in FILE_SUFFIX.items():
        path = results_dir / f"{stem}_{suffix}.json"
        out[k] = _metric(_read_final(path), metric)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--out", type=str, default="benchmarks/ablation_model_components.json")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    series = {
        "Logic (ACD)": _load_series(results_dir, "logic", "acd"),
        "M5 (RMSSE)": _load_series(results_dir, "m5", "rmsse"),
    }
    payload = {
        "mode": "measured",
        "higher_is_better": False,
        "metric_name": "Logic ACD / M5 RMSSE (lower is better; bars show performance drop %)",
        "series": series,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    for name, scores in series.items():
        full = scores["full"]
        print(f"  {name}: full={full:.4f}")
        for k in ("wo_hyp", "wo_acg", "wo_ph_scan"):
            v = scores[k]
            drop = 100.0 * (v - full) / full if abs(full) > 1e-12 else 0.0
            print(f"    {k}: {v:.4f}  drop={drop:+.1f}%")


if __name__ == "__main__":
    main()
