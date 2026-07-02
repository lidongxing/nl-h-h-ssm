from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple


MetricName = Literal["smape", "rmsse", "acd"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    latex_name: str
    group: Literal["A", "B", "C"]
    metric: MetricName


# Table 5 structure (edit keys to match your dataset stems in results/*.json)
TABLE5_DATASETS: List[DatasetSpec] = [
    DatasetSpec(key="m5", latex_name="M5", group="A", metric="smape"),
    DatasetSpec(key="wiki", latex_name="Wiki", group="A", metric="smape"),
    DatasetSpec(key="traffic", latex_name="Traffic", group="B", metric="rmsse"),
    DatasetSpec(key="elect", latex_name="Elect.", group="B", metric="rmsse"),
    DatasetSpec(key="exp-log", latex_name="Exp-Log", group="C", metric="acd"),
    DatasetSpec(key="diag", latex_name="Diag.", group="C", metric="acd"),
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _extract_final_metrics(payload: dict) -> Dict[str, float]:
    """
    Supports the structure produced by experiments/run_exp.py:
      { ..., "final": {"smape":..., "rmsse":..., "acd":...}, ...}
    Also supports flat top-level keys as fallback.
    """
    out: Dict[str, float] = {}
    if isinstance(payload.get("final"), dict):
        for k in ("smape", "rmsse", "acd"):
            v = payload["final"].get(k)
            if isinstance(v, (int, float)) and math.isfinite(v):
                out[k] = float(v)
    for k in ("smape", "rmsse", "acd"):
        if k not in out:
            v = payload.get(k)
            if isinstance(v, (int, float)) and math.isfinite(v):
                out[k] = float(v)
    return out


def _extract_model_name(payload: dict, filename: str) -> str:
    m = payload.get("model")
    if isinstance(m, str) and m:
        return m
    # fallback: infer from filename like {dataset}_{model}.json
    stem = Path(filename).stem
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


def _extract_dataset_key(payload: dict, filename: str) -> str:
    d = payload.get("dataset")
    if isinstance(d, str) and d:
        return d
    stem = Path(filename).stem
    if "_" in stem:
        return "_".join(stem.split("_")[:-1])
    return stem


def _format_cell(v: Optional[float], ndigits: int) -> str:
    if v is None:
        return "-"
    fmt = f"{{:.{ndigits}f}}"
    return fmt.format(v)


def _latex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def _rank_per_column(values_by_model: Dict[str, Optional[float]]) -> Dict[str, int]:
    # Lower is better; missing -> worst rank at end
    present = [(m, v) for m, v in values_by_model.items() if v is not None and math.isfinite(v)]
    present.sort(key=lambda x: x[1])  # lower better
    ranks: Dict[str, int] = {}
    for i, (m, _v) in enumerate(present, start=1):
        ranks[m] = i
    worst = len(present) + 1
    for m, v in values_by_model.items():
        if m not in ranks:
            ranks[m] = worst
    return ranks


def _best_and_second(values_by_model: Dict[str, Optional[float]]) -> Tuple[Optional[str], Optional[str]]:
    present = [(m, v) for m, v in values_by_model.items() if v is not None and math.isfinite(v)]
    present.sort(key=lambda x: x[1])
    best = present[0][0] if len(present) >= 1 else None
    second = present[1][0] if len(present) >= 2 else None
    return best, second


def generate_table(
    results_dir: Path,
    dataset_specs: List[DatasetSpec],
    *,
    ndigits: int = 3,
    improvement_ref_model: str = "mamba-2",
) -> str:
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No json files found in {results_dir}")

    # data[(model, dataset_key)] = metrics dict
    data: Dict[Tuple[str, str], Dict[str, float]] = {}
    models: List[str] = []
    datasets_seen: List[str] = []

    for fp in files:
        payload = _read_json(fp)
        model = _extract_model_name(payload, fp.name)
        dkey = _extract_dataset_key(payload, fp.name)
        metrics = _extract_final_metrics(payload)
        data[(model, dkey)] = metrics
        if model not in models:
            models.append(model)
        if dkey not in datasets_seen:
            datasets_seen.append(dkey)

    # Build column values per dataset spec
    col_values: Dict[str, Dict[str, Optional[float]]] = {}  # col_key -> model -> val
    for spec in dataset_specs:
        col_key = spec.key
        col_values[col_key] = {}
        for model in models:
            m = data.get((model, spec.key), {})
            col_values[col_key][model] = m.get(spec.metric)

    # Rank per dataset column
    ranks_per_col: Dict[str, Dict[str, int]] = {k: _rank_per_column(vs) for k, vs in col_values.items()}
    avg_rank: Dict[str, float] = {}
    for model in models:
        rs = [ranks_per_col[spec.key][model] for spec in dataset_specs]
        avg_rank[model] = sum(rs) / len(rs)

    # Improvement: relative reduction in avg_rank vs reference model (if present)
    impr: Dict[str, Optional[float]] = {m: None for m in models}
    ref = None
    for m in models:
        if m.lower() == improvement_ref_model.lower():
            ref = m
            break
    if ref is not None:
        ref_rank = avg_rank[ref]
        for m in models:
            if m == ref:
                impr[m] = None
            else:
                impr[m] = (ref_rank - avg_rank[m]) / max(1e-9, ref_rank) * 100.0

    # Best/second per dataset column for styling
    best_second: Dict[str, Tuple[Optional[str], Optional[str]]] = {k: _best_and_second(vs) for k, vs in col_values.items()}

    # Build LaTeX table (Table 5-like)
    # Column layout: Model | GroupA(2) | GroupB(2) | GroupC(2) | Summary(2)
    cols_a = [s for s in dataset_specs if s.group == "A"]
    cols_b = [s for s in dataset_specs if s.group == "B"]
    cols_c = [s for s in dataset_specs if s.group == "C"]

    header = []
    header.append(r"\begin{table}[t]")
    header.append(r"\centering")
    header.append(r"\small")
    header.append(r"\setlength{\tabcolsep}{6pt}")
    header.append(r"\begin{tabular}{l|" + "cc|cc|cc|cc}")
    header.append(r"\hline")
    header.append(
        r"\textbf{Model} & \multicolumn{2}{c|}{\textbf{Group A (Economic)}} & "
        r"\multicolumn{2}{c|}{\textbf{Group B (Industrial)}} & "
        r"\multicolumn{2}{c|}{\textbf{Group C (Expert)}} & "
        r"\multicolumn{2}{c}{\textbf{Summary}} \\"
    )
    header.append(
        " & "
        + " & ".join([f"\\textbf{{{s.latex_name}}}" for s in (cols_a + cols_b + cols_c)])
        + r" & \textbf{Avg. Rank} & \textbf{Imprv. \%} \\"
    )
    header.append(r"\hline")

    # Sort models by avg rank (best first)
    models_sorted = sorted(models, key=lambda m: avg_rank[m])

    lines = []
    for model in models_sorted:
        row = [f"\\textbf{{{_latex_escape(model)}}}" if model.lower() == "nlh_ssm" else _latex_escape(model)]
        for spec in (cols_a + cols_b + cols_c):
            v = col_values[spec.key][model]
            cell = _format_cell(v, ndigits)
            best, second = best_second[spec.key]
            if best == model:
                cell = f"\\textbf{{{cell}}}"
            elif second == model:
                cell = f"\\underline{{{cell}}}"
            row.append(cell)
        row.append(f"{avg_rank[model]:.1f}")
        if impr[model] is None:
            row.append("-")
        else:
            sign = "+" if impr[model] >= 0 else ""
            row.append(f"{sign}{impr[model]:.1f}\\%")
        lines.append(" & ".join(row) + r" \\")

    footer = []
    footer.append(r"\hline")
    footer.append(r"\end{tabular}")
    footer.append(r"\caption{Main results on hierarchical datasets. Lower is better. Bold: best. Underline: second best.}")
    footer.append(r"\label{tab:main_results}")
    footer.append(r"\end{table}")

    return "\n".join(header + lines + footer) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results", help="Folder containing per-run JSON result files")
    ap.add_argument("--out", type=str, default="table5.tex", help="Output .tex file path")
    ap.add_argument("--ndigits", type=int, default=3, help="Digits after decimal for metric cells")
    ap.add_argument("--impr_ref", type=str, default="mamba-2", help="Reference model name for Imprv. %")
    args = ap.parse_args()

    tex = generate_table(
        Path(args.results_dir),
        TABLE5_DATASETS,
        ndigits=args.ndigits,
        improvement_ref_model=args.impr_ref,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex, encoding="utf-8")
    print(f"Wrote LaTeX table to {out_path}")


if __name__ == "__main__":
    main()
