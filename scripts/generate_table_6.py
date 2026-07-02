"""
Build LaTeX for **Table 6** (ten hierarchical datasets × baselines + NL-H-H-SSM).

Expects ``results/*.json`` files shaped like ``experiments/run_exp.py`` outputs:
``{"model": "...", "dataset": "m5", "final": {"smape":..., "rmsse":..., "acd":...}}``.

Metrics per column follow the paper layout:
  - Socio-economic (Tour., Lab., Pris.): sMAPE
  - Industrial \& IoT (M5, Wiki, Elec., Traf., Solr.): RMSSE
  - Expert systems (Logic., Med.): ACD

Lower is better. Bold = best; underline = second best (uses ``booktabs``).

CLI: ``--format harv`` emits the two-row Table~6 block used in ``elsarticle-template-harv.tex``;
``--format simple`` emits a compact standalone ``table*`` (label ``tab:main_results_ten``).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse ranking helpers from Table 5 generator
from benchmarks.generate_table import (
    _best_and_second,
    _extract_dataset_key,
    _extract_model_name,
    _format_cell,
    _latex_escape,
    _rank_per_column,
    _read_json,
)

MetricName = Literal["smape", "rmsse", "acd"]


@dataclass(frozen=True)
class Table6DatasetSpec:
    key: str
    latex_name: str
    group: Literal["SE", "IND", "EXP"]
    metric: MetricName


TABLE6_DATASETS: List[Table6DatasetSpec] = [
    Table6DatasetSpec("tourism", "Tour.", "SE", "smape"),
    Table6DatasetSpec("labor", "Lab.", "SE", "smape"),
    Table6DatasetSpec("prison", "Pris.", "SE", "smape"),
    Table6DatasetSpec("m5", "M5", "IND", "rmsse"),
    Table6DatasetSpec("wiki", "Wiki", "IND", "rmsse"),
    Table6DatasetSpec("electricity", "Elec.", "IND", "rmsse"),
    Table6DatasetSpec("traffic", "Traf.", "IND", "rmsse"),
    Table6DatasetSpec("solar", "Solr.", "IND", "rmsse"),
    Table6DatasetSpec("logic", "Logic.", "EXP", "acd"),
    Table6DatasetSpec("medical", "Med.", "EXP", "acd"),
]


def _extract_final_metrics(payload: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(payload.get("final"), dict):
        for k in ("smape", "rmsse", "acd", "tw_mse", "crps"):
            v = payload["final"].get(k)
            if isinstance(v, (int, float)) and math.isfinite(v):
                out[k] = float(v)
    for k in ("smape", "rmsse", "acd", "tw_mse", "crps"):
        if k not in out:
            v = payload.get(k)
            if isinstance(v, (int, float)) and math.isfinite(v):
                out[k] = float(v)
    return out


def generate_table6_tex(
    results_dir: Path,
    *,
    ndigits: int = 3,
    improvement_ref_model: str = "mamba-2",
) -> str:
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No json files found in {results_dir}")

    data: Dict[Tuple[str, str], Dict[str, float]] = {}
    models: List[str] = []

    for fp in files:
        payload = _read_json(fp)
        model = _extract_model_name(payload, fp.name)
        dkey = _extract_dataset_key(payload, fp.name)
        metrics = _extract_final_metrics(payload)
        data[(model, dkey)] = metrics
        if model not in models:
            models.append(model)

    col_values: Dict[str, Dict[str, Optional[float]]] = {}
    for spec in TABLE6_DATASETS:
        col_values[spec.key] = {}
        for model in models:
            m = data.get((model, spec.key), {})
            col_values[spec.key][model] = m.get(spec.metric)

    ranks_per_col = {k: _rank_per_column(vs) for k, vs in col_values.items()}
    avg_rank: Dict[str, float] = {}
    for model in models:
        rs = [ranks_per_col[spec.key][model] for spec in TABLE6_DATASETS]
        avg_rank[model] = sum(rs) / len(rs)

    impr: Dict[str, Optional[float]] = {m: None for m in models}
    ref = None
    for m in models:
        if m.lower().replace("_", "-") == improvement_ref_model.lower().replace("_", "-"):
            ref = m
            break
    if ref is not None:
        ref_rank = avg_rank[ref]
        for m in models:
            if m == ref:
                impr[m] = None
            else:
                impr[m] = (ref_rank - avg_rank[m]) / max(1e-9, ref_rank) * 100.0

    best_second = {k: _best_and_second(vs) for k, vs in col_values.items()}

    se = [s for s in TABLE6_DATASETS if s.group == "SE"]
    ind = [s for s in TABLE6_DATASETS if s.group == "IND"]
    exp = [s for s in TABLE6_DATASETS if s.group == "EXP"]
    ordered = se + ind + exp

    ncols = len(ordered) + 2
    colspec = "l|" + "c" * len(se) + "|" + "c" * len(ind) + "|" + "c" * len(exp) + "|cc"

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \multicolumn{"
        + str(len(se))
        + r"}{c|}{\textbf{Socio-Economic (sMAPE $\downarrow$)}} & \multicolumn{"
        + str(len(ind))
        + r"}{c|}{\textbf{Industrial \& IoT (RMSSE $\downarrow$)}} & \multicolumn{"
        + str(len(exp))
        + r"}{c|}{\textbf{Expert Systems (ACD $\downarrow$)}} & \textbf{Avg. Rank} & \textbf{Impv. \%} \\"
    )
    lines.append(
        " & "
        + " & ".join(f"\\textbf{{{s.latex_name}}}" for s in ordered)
        + r" & & \\"
    )
    lines.append(r"\midrule")

    models_sorted = sorted(models, key=lambda m: avg_rank[m])
    for model in models_sorted:
        disp = "\\textbf{NL-H-H-SSM (Proposed)}" if model.lower() in {"nlh_ssm", "nlh-h-h-ssm"} else _latex_escape(model)
        row = [disp]
        for spec in ordered:
            v = col_values[spec.key][model]
            cell = _format_cell(v, ndigits)
            best, second = best_second[spec.key]
            if best == model:
                cell = f"\\textbf{{{cell}}}"
            elif second == model:
                cell = f"\\underline{{{cell}}}"
            row.append(cell)
        row.append(f"{avg_rank[model]:.1f}")
        iv = impr[model]
        if iv is None:
            row.append("-")
        else:
            sign = "+" if iv >= 0 else ""
            row.append(f"{sign}{iv:.1f}\\%")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Main results on ten real-world hierarchical datasets. "
        r"Metrics: sMAPE (socio-economic), RMSSE (industrial/IoT), ACD (expert systems). "
        r"Lower is better. \textbf{Bold}: best; \underline{underline}: second best.}"
    )
    lines.append(r"\label{tab:main_results_ten}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


PAPER_MODEL_ORDER = ("transformer", "informer", "mamba1", "mamba2", "nlh_ssm")

MODEL_DISPLAY = {
    "transformer": "Transformer",
    "informer": "Informer",
    "mamba1": "Mamba-1",
    "mamba2": "Mamba-2",
    "nlh_ssm": "\\textbf{NL-H-H-SSM}",
}


def generate_table6_harv_tex(
    results_dir: Path,
    *,
    ndigits: int = 3,
    improvement_ref_model: str = "mamba2",
) -> str:
    """
    ``elsarticle-template-harv.tex`` style Table~6: two-line header, ``booktabs``, fixed baseline order,
    gray row for NL-H-H-SSM. Uses ``results/*.json`` from ``run_exp.py``.
    """
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No json files found in {results_dir}")

    data: Dict[Tuple[str, str], Dict[str, float]] = {}
    models: List[str] = []
    for fp in files:
        payload = _read_json(fp)
        model = _extract_model_name(payload, fp.name)
        dkey = _extract_dataset_key(payload, fp.name)
        metrics = _extract_final_metrics(payload)
        data[(model, dkey)] = metrics
        if model not in models:
            models.append(model)

    col_values: Dict[str, Dict[str, Optional[float]]] = {}
    for spec in TABLE6_DATASETS:
        col_values[spec.key] = {}
        for model in models:
            m = data.get((model, spec.key), {})
            col_values[spec.key][model] = m.get(spec.metric)

    ranks_per_col = {k: _rank_per_column(vs) for k, vs in col_values.items()}
    avg_rank: Dict[str, float] = {}
    for model in models:
        rs = [ranks_per_col[spec.key][model] for spec in TABLE6_DATASETS]
        avg_rank[model] = sum(rs) / len(rs)

    impr: Dict[str, Optional[float]] = {m: None for m in models}
    ref = None
    for m in models:
        if m.lower().replace("_", "-") == improvement_ref_model.lower().replace("_", "-"):
            ref = m
            break
    if ref is not None:
        ref_rank = avg_rank[ref]
        for m in models:
            if m == ref:
                impr[m] = None
            else:
                impr[m] = (ref_rank - avg_rank[m]) / max(1e-9, ref_rank) * 100.0

    best_second = {k: _best_and_second(vs) for k, vs in col_values.items()}
    se = [s for s in TABLE6_DATASETS if s.group == "SE"]
    ind = [s for s in TABLE6_DATASETS if s.group == "IND"]
    exp = [s for s in TABLE6_DATASETS if s.group == "EXP"]
    ordered = se + ind + exp

    row_models: List[str] = [m for m in PAPER_MODEL_ORDER if m in models]
    for m in sorted(models, key=lambda x: avg_rank[x]):
        if m not in row_models:
            row_models.append(m)

    lines: List[str] = []
    lines.append(r"\begin{table*}[!tbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Main Results on Ten Real-World Hierarchical Datasets. Predictive accuracy is measured by "
        r"sMAPE (Socio-Economic) and RMSSE (Industrial and IoT), while structural fidelity is quantified by ACD "
        r"(Expert Systems). $\downarrow$ indicates lower values are better; $\uparrow$ indicates higher values are "
        r"better. \textbf{bold}: best; \underline{underline}: second best.}"
    )
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{l|ccc|ccccc|cc|c}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Category} & \multicolumn{3}{c|}{\textbf{Socio-Economic}}"
        r" & \multicolumn{5}{c|}{\textbf{Industrial \& IoT}}"
        r" & \multicolumn{2}{c|}{\textbf{Expert Systems}}"
        r" & \textbf{Summary} \\"
    )
    lines.append(r"\midrule")
    lines.append(
        r"\textbf{Model} & "
        + " & ".join(f"\\textbf{{{s.latex_name}}}" for s in ordered)
        + r" & \textbf{Avg. Rank} \\"
    )
    lines.append(r"\midrule")

    for model in row_models:
        is_proposed = model.lower() in {"nlh_ssm", "nlh-h-h-ssm"}
        if is_proposed:
            lines.append(r"\midrule")
            lines.append(r"\rowcolor[gray]{0.95}")
        disp = MODEL_DISPLAY.get(model, _latex_escape(model))
        row = [disp]
        for spec in ordered:
            v = col_values[spec.key][model]
            cell = _format_cell(v, ndigits)
            best, second = best_second[spec.key]
            if best == model:
                cell = f"\\textbf{{{cell}}}"
            elif second == model:
                cell = f"\\underline{{{cell}}}"
            row.append(cell)
        row.append(f"{avg_rank[model]:.1f}")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--out", type=str, default="table6.tex")
    ap.add_argument("--ndigits", type=int, default=3)
    ap.add_argument("--impr_ref", type=str, default="mamba2")
    ap.add_argument(
        "--format",
        choices=("simple", "harv"),
        default="simple",
        help="simple: compact standalone table; harv: elsarticle two-line header + booktabs (Table 6).",
    )
    args = ap.parse_args()
    rd = Path(args.results_dir)
    if args.format == "harv":
        tex = generate_table6_harv_tex(
            rd,
            ndigits=args.ndigits,
            improvement_ref_model=args.impr_ref,
        )
    else:
        tex = generate_table6_tex(
            rd,
            ndigits=args.ndigits,
            improvement_ref_model=args.impr_ref,
        )
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(tex, encoding="utf-8")
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
