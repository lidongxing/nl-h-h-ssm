"""
Validate Table 3 dataset statistics against expected values.

Supported input formats:
  - CSV with headers including at least:
      Dataset, Layers, Total Series, Avg. Branching Factor
    Optional: Domain
  - JSON as a list of row objects with the same keys.

Example:
  python scripts/validate_table3_stats.py --input table3.csv
  python scripts/validate_table3_stats.py --input table3.json --skip-domain
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ExpectedRow:
    domain: str
    layers: int
    total_series_raw: str
    avg_branching_factor: float


EXPECTED: Dict[str, ExpectedRow] = {
    "tourism-au": ExpectedRow("Tourism", 4, "555", 7.5),
    "labour-au": ExpectedRow("Economy", 3, "128", 4.0),
    "prison-au": ExpectedRow("Social", 3, "81", 3.0),
    "m5-walmart": ExpectedRow("Retail", 12, "30490", 10.2),
    "wiki-traffic": ExpectedRow("Web", 5, "145063", 125.0),
    "electricity-l": ExpectedRow("Energy", 3, "370", 18.0),
    "traffic-hts": ExpectedRow("Transport", 4, "963", 24.5),
    "solar-hts": ExpectedRow("Energy", 5, "137", 5.2),
    "logicgraph": ExpectedRow("Expert", 20, "10240+", 2.0),
    "med-diag-path": ExpectedRow("Medical", 15, "5500", 3.5),
}


def _norm_key(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _parse_total(raw: str) -> tuple[int, bool]:
    s = raw.strip().replace(",", "")
    has_plus = s.endswith("+")
    if has_plus:
        s = s[:-1]
    return int(s), has_plus


def _parse_float(raw: str) -> float:
    return float(raw.strip().replace(",", ""))


def _load_rows(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of row objects.")
        return [dict(x) for x in data]
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _get_field(row: dict, names: List[str]) -> Optional[str]:
    lowered = {str(k).strip().lower(): str(v) for k, v in row.items() if k is not None}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def validate(rows: List[dict], check_domain: bool = True, bf_tol: float = 1e-6) -> int:
    errors: List[str] = []
    seen: set[str] = set()

    for row in rows:
        ds = _get_field(row, ["dataset", "data set"])
        if not ds:
            continue
        key = _norm_key(ds)
        if key not in EXPECTED:
            continue
        seen.add(key)
        exp = EXPECTED[key]

        layers_s = _get_field(row, ["layers"])
        total_s = _get_field(row, ["total series", "total_series"])
        bf_s = _get_field(row, ["avg. branching factor", "avg branching factor", "bf"])
        dom_s = _get_field(row, ["domain"])

        if layers_s is None or total_s is None or bf_s is None:
            errors.append(f"[{ds}] missing required columns (Layers/Total Series/BF).")
            continue

        # Layers
        try:
            layers = int(layers_s.replace(",", "").strip())
            if layers != exp.layers:
                errors.append(f"[{ds}] Layers mismatch: got {layers}, expected {exp.layers}.")
        except Exception:
            errors.append(f"[{ds}] Layers parse error: {layers_s!r}.")

        # Total Series
        try:
            got_total, got_plus = _parse_total(total_s)
            exp_total, exp_plus = _parse_total(exp.total_series_raw)
            if exp_plus:
                # Expected is lower-bound style, e.g., "10240+"
                if got_total < exp_total:
                    errors.append(
                        f"[{ds}] Total Series too small: got {got_total}, expected >= {exp_total}."
                    )
            else:
                if got_total != exp_total:
                    errors.append(
                        f"[{ds}] Total Series mismatch: got {got_total}, expected {exp_total}."
                    )
            # If expected has plus but observed does not, allow it (still valid exact/lower bound).
            # If expected has no plus but observed has plus, treat as mismatch.
            if (not exp_plus) and got_plus:
                errors.append(f"[{ds}] Total Series uses '+', but expected exact value.")
        except Exception:
            errors.append(f"[{ds}] Total Series parse error: {total_s!r}.")

        # BF
        try:
            bf = _parse_float(bf_s)
            if not math.isfinite(bf) or abs(bf - exp.avg_branching_factor) > bf_tol:
                errors.append(
                    f"[{ds}] Avg. Branching Factor mismatch: got {bf}, expected {exp.avg_branching_factor}."
                )
        except Exception:
            errors.append(f"[{ds}] Avg. Branching Factor parse error: {bf_s!r}.")

        # Domain (optional check)
        if check_domain:
            if dom_s is None:
                errors.append(f"[{ds}] Domain column missing.")
            elif dom_s.strip() != exp.domain:
                errors.append(f"[{ds}] Domain mismatch: got {dom_s!r}, expected {exp.domain!r}.")

    missing = sorted(set(EXPECTED.keys()) - seen)
    if missing:
        errors.append("Missing datasets in input: " + ", ".join(missing))

    if errors:
        print("Validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("Validation PASSED for all 10 datasets.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Table 3 statistics.")
    ap.add_argument("--input", required=True, help="Path to table3 CSV/JSON file.")
    ap.add_argument(
        "--skip-domain",
        action="store_true",
        help="Skip Domain column validation.",
    )
    ap.add_argument(
        "--bf-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for Avg. Branching Factor comparison.",
    )
    args = ap.parse_args()

    rows = _load_rows(Path(args.input))
    code = validate(rows, check_domain=not args.skip_domain, bf_tol=args.bf_tol)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

