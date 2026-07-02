#!/usr/bin/env python3
"""Generate REPO_MANIFEST.json with a unique description for every tracked file."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Static descriptions (unique per path; no copy-paste templates) ---
STATIC: dict[str, str] = {
    "README.md": "Project overview, install steps, Table 6 / RQ2 / RQ3 reproduction, link to FILE_INDEX.",
    "FILE_INDEX.md": "Human-readable catalog: one distinct comment per tracked path.",
    "REPO_MANIFEST.json": "Machine-readable catalog (this file): path → unique description for reviewers and scripts.",
    "setup.py": "Setuptools package `nlh-ssm`; core deps torch/pyyaml/pandas/numpy<2; optional [mamba] for baselines.",
    ".gitignore": "Omits raw data, four CSVs >100 MB, __pycache__, egg-info, generated PNG/PDF under assets/.",
    ".gitattributes": "Normalizes shell scripts to LF for Linux reproduction after Windows checkout.",
    "assets/.gitkeep": "Reserves assets/ for benchmark-generated figures (PNG not committed).",
    "assets/README.md": "Lists which script produces figure7/9/10 and expected output filenames.",
    "nlh_ssm/__init__.py": "Top-level import namespace for the NL-H-H-SSM Python package.",
    "nlh_ssm/README.md": "Index of hyperbolic ops, NL-H modules, loaders, and metric subpackages.",
    "nlh_ssm/ops/hyperbolic.py": "Poincaré exp₀/log₀, Möbius addition, distance, projection — Eqs. (3)–(5) in paper.",
    "nlh_ssm/ops/check_nan.py": "Guards hyperbolic forward/backward against NaN/Inf during long Table 6 runs.",
    "nlh_ssm/modules/nlh_block.py": "Single NL-H layer: hyperbolic state, PH-Scan, ACG; ablation switches wo_hyp/wo_acg/wo_ph_scan.",
    "nlh_ssm/modules/hierarchy_embedding.py": "Maps slash-separated series IDs to depth features feeding ACG curvature.",
    "nlh_ssm/models/mixer_seq_simple.py": "End-to-end sequence model (input proj → NL-H stack → head); Table 6 & speed_test entry.",
    "nlh_ssm/models/selective_state_update.py": "Selective SSM gate compatible with Mamba-style interfaces inside NL-H.",
    "nlh_ssm/data/__init__.py": "Exports get_dataloader for experiments and benchmarks.",
    "nlh_ssm/data/loader.py": "Long-format CSV windows; infers hierarchy from id delimiter for ACD evaluation.",
    "nlh_ssm/metrics/__init__.py": "Exports forecast and structure metric functions.",
    "nlh_ssm/metrics/accuracy.py": "sMAPE (Tourism/Labour/Prison), RMSSE (M5–Solar), TW-MSE, CRPS implementations.",
    "nlh_ssm/metrics/distance.py": "ACD and MAT for LogicGraph and Med-Diag-Path expert-system columns.",
    "nlh_ssm/utils/__init__.py": "Shared utility exports for metrics submodule.",
    "nlh_ssm/utils/metrics.py": "Numerically stable kernels reused by accuracy.py and distance.py.",
    "csrc/README.md": "Documents PH-Scan kernel and Cauchy helper modules.",
    "csrc/ph_scan_kernel.py": "Manifold prefix scan O(L/M+log M); Triton path or PyTorch fallback (§4 PH-Scan).",
    "csrc/cauchy.py": "Cauchy/resolvent helpers stabilizing hyperbolic scan accumulation.",
    "models/README.md": "Facade layer between run_exp.py and nlh_ssm internals.",
    "models/__init__.py": "Registers nlh_ssm, ph_scan, acg_gate model names for CLI.",
    "models/nlh_ssm.py": "run_exp-facing wrapper: nlh_num_layers, nlh_expand, nlh_c_base from yaml.",
    "models/ph_scan.py": "Thin PH-Scan module for tests and operator ablations.",
    "models/acg_gate.py": "Adaptive Curvature Gate: depth-dependent c_eff from hierarchy metadata.",
    "experiments/README.md": "Training CLIs, device launcher, and per-dataset YAML examples.",
    "experiments/__init__.py": "Setuptools namespace for experiments package.",
    "experiments/.gitkeep": "Historical placeholder before configs were added.",
    "experiments/run_exp.py": "Main trainer CLI: 5 models × 1 CSV → results/{stem}_{model}.json + metric history.",
    "experiments/run_main.py": "Loads experiments/configs/*.yaml and delegates to run_exp.",
    "experiments/launch_device.py": "Resolves device: cpu | cuda:N | multi-GPU DDP list from YAML.",
    "experiments/train_all.sh": "Sequential launcher over all experiments/configs/*.yaml.",
    "configs/README.md": "Explains nlh_tuned.yaml fields used in fair Table 6 comparison.",
    "configs/nlh_tuned.yaml": "Paper windows & NL-H LR/clip/c_base/layers per stem (tourism…medical).",
    "configs/nlh_tuned.example.yaml": "Empty schema sample for tune_nlh_ssm.py output.",
    "data/README.md": "Download → preprocess pipeline; which CSVs are committed vs rebuilt.",
    "data/.gitkeep": "Marks data/ root in git.",
    "data/download_all.sh": "curl/wget ABS + Zenodo archives; documents manual Kaggle M5 step.",
    "data/preprocess.py": "Raw zips/xlsx/LogicGraph → unified id/time/value CSVs for loader.",
    "data/raw/.gitkeep": "Raw downloads directory (contents gitignored).",
    "data/processed/.gitkeep": "Processed CSV output directory marker.",
    "data/processed/tourism.csv": "ABS tourism satellite account series; 4-layer hierarchy; sMAPE column Tour.",
    "data/processed/labor.csv": "ABS labour account table-1; 3-layer; sMAPE column Lab.",
    "data/processed/prison.csv": "ABS corrective services Jun-2023; 3-layer; sMAPE column Pris.",
    "data/processed/wiki.csv": "Monash Kaggle web traffic HTS; 5-layer; RMSSE column Wiki.",
    "data/processed/logic.csv": "LogicGraph-derived expert traces; deep hierarchy; ACD column Logic.",
    "data/processed/medical.csv": "Med-Diag-Path synthetic diagnostic hierarchies; ACD column Med.",
    "scripts/README.md": "Table 6 campaign, LaTeX export, tuning, RQ3 orchestration scripts.",
    "scripts/run_table6_all.sh": "Default: 10 datasets × 5 models @ 40 epochs; subcommands rank/tex/nlh_only.",
    "scripts/run_table6_fair_baselines.sh": "Baselines only, same seq_len/stride as nlh_tuned.yaml, lr=0.001.",
    "scripts/table6_nlh_campaign.sh": "Optional tune phases A/B then train for NL-H per dataset.",
    "scripts/generate_table_6.py": "JSON → elsarticle Table 6 LaTeX (harv format, no Impv. column).",
    "scripts/rank_table6.py": "Prints per-dataset ranks and mean rank — quick audit of results/.",
    "scripts/validate_table3_stats.py": "Asserts layer count & branching vs paper Table 3 expected rows.",
    "scripts/generate_repo_manifest.py": "Builds REPO_MANIFEST.json and docs/descriptions/*.md — one unique sentence per tracked path.",
    "scripts/upload_from_manifest.py": "Commits one file per REPO_MANIFEST entry; each commit message = that file's unique description.",
    "scripts/commit_doc_layers.sh": "Optional helper: batch-commit docs/descriptions/ by top-level folder with distinct messages.",
    "docs/README.md": "Explains docs/descriptions/ stubs and GitHub last-commit-message vs per-file notes.",
    "scripts/tune_nlh_ssm.py": "Validation-grid search; writes overrides into nlh_tuned.yaml.",
    "scripts/run_speed_forward_bench.sh": "Invokes speed_test.py --forward-only → speed_bench.json + Fig.7.",
    "scripts/run_speed_training_bench.sh": "Full fwd+bwd training benchmark → speed_bench_train.json.",
    "scripts/run_rq3_ablation.sh": "Logic+M5 runs with wo_hyp, wo_acg, wo_ph_scan flags.",
    "scripts/run_parameter_sensitivity.sh": "Logic sweeps: nlh_c_base, nlh_expand, mamba2 d_state.",
    "scripts/run_rq3_enhancements_all.sh": "Chains ablation → sensitivity → training speed (RQ3).",
    "scripts/aggregate_component_ablation.py": "Collates ablation runs into ablation_model_components.json.",
    "scripts/aggregate_parameter_sensitivity.py": "Collates sweeps into figure10_parameter_sensitivity.json.",
    "benchmarks/README.md": "Speed, ablation JSON, sensitivity raw runs, and figure generators.",
    "benchmarks/__init__.py": "Benchmark package marker for setuptools.",
    "benchmarks/speed_test.py": "A800 protocol: NL-H vs Mamba-2 vs Transformer @ L=1k…128k, d=768.",
    "benchmarks/speed_results/speed_bench.json": "Committed forward-only tokens/s & VRAM — paper Fig.7 / §6.2.",
    "benchmarks/speed_results/speed_bench_train.json": "Committed fwd+bwd training throughput — paper Table 7.",
    "benchmarks/speed_results/speed_bench_synthetic.json": "Short synthetic-length smoke timings for speed_test.py.",
    "benchmarks/ablation_model_components.json": "RQ3 bar-chart source: Hyp / PH-Scan / ACG removal on Logic+M5.",
    "benchmarks/ablation_operator.json": "Euclidean vs hyperbolic scan operator comparison summary.",
    "benchmarks/figure10_parameter_sensitivity.json": "Aggregated Logic c_base & capacity sensitivity curves.",
    "benchmarks/model_ablation_figure9.py": "Renders assets/figure9_component_ablation.png from ablation JSON.",
    "benchmarks/parameter_sensitivity_figure10.py": "Renders assets/figure10_parameter_sensitivity.png.",
    "benchmarks/operator_ablation.py": "Driver for operator-level ablation experiments.",
    "benchmarks/generate_table.py": "Legacy six-dataset LaTeX table from older results layout.",
    "benchmarks/plot_radar.py": "Radar plot comparing models across normalized metrics.",
    "benchmarks/plot_curvature_heatmap.py": "Schematic curvature/regime heatmap for paper illustration.",
    "benchmarks/plot_poincare_vs_euclidean.py": "Schematic Poincaré vs Euclidean embedding figure.",
    "benchmarks/figure_architecture_english.py": "English NL-H-H-SSM block diagram for manuscript.",
    "tests/README.md": "pytest entry points for hyperbolic, PH-Scan, and metrics.",
    "tests/test_hyperbolic.py": "Round-trip exp/log and Möbius identity checks on Poincaré ball.",
    "tests/test_ph_scan.py": "PH-Scan output shape, dtype, and finite-value smoke test.",
    "tests/test_metrics.py": "Hand-checked sMAPE, RMSSE, ACD against reference values.",
    "results/README.md": "All 50 Table 6 JSON logs with final sMAPE/RMSSE/ACD per file.",
}

EXPERIMENT_CONFIGS = {
    "experiments/configs/tourism.yaml": "Single-run example: Tourism-AU CSV, default epochs, optional cuda device.",
    "experiments/configs/labor.yaml": "Single-run example: Labour-AU CSV path and training length.",
    "experiments/configs/prison.yaml": "Single-run example: Prison-AU CSV path and training length.",
    "experiments/configs/m5.yaml": "Single-run example: M5-Walmart (large CSV rebuilt locally).",
    "experiments/configs/wiki.yaml": "Single-run example: Wiki-Traffic Monash HTS CSV.",
    "experiments/configs/electricity.yaml": "Single-run example: Electricity-L hourly HTS CSV.",
    "experiments/configs/traffic.yaml": "Single-run example: Traffic-HTS hourly CSV.",
    "experiments/configs/solar.yaml": "Single-run example: Solar-HTS 10-minute CSV.",
    "experiments/configs/logic.yaml": "Single-run example: LogicGraph ACD evaluation CSV.",
    "experiments/configs/medical.yaml": "Single-run example: Med-Diag-Path ACD evaluation CSV.",
}

SENSITIVITY = {
    "benchmarks/sensitivity_results/logic_cbase_0.1.json": "Logic sweep: nlh_c_base=0.1 — low curvature, ACD plateau check.",
    "benchmarks/sensitivity_results/logic_cbase_0.5.json": "Logic sweep: nlh_c_base=0.5 — mid-low curvature.",
    "benchmarks/sensitivity_results/logic_cbase_1.0.json": "Logic sweep: nlh_c_base=1.0 — nominal curvature setting.",
    "benchmarks/sensitivity_results/logic_cbase_1.5.json": "Logic sweep: nlh_c_base=1.5 — elevated curvature.",
    "benchmarks/sensitivity_results/logic_cbase_2.0.json": "Logic sweep: nlh_c_base=2.0 — high curvature upper sweep.",
    "benchmarks/sensitivity_results/logic_expand_1.json": "Logic sweep: nlh_expand=1 — minimal inner width.",
    "benchmarks/sensitivity_results/logic_expand_2.json": "Logic sweep: nlh_expand=2 — paper default width.",
    "benchmarks/sensitivity_results/logic_expand_4.json": "Logic sweep: nlh_expand=4 — 2× default capacity.",
    "benchmarks/sensitivity_results/logic_expand_8.json": "Logic sweep: nlh_expand=8 — 4× default capacity.",
    "benchmarks/sensitivity_results/logic_mamba2_dstate_16.json": "Logic baseline: Mamba-2 with d_state=16 for capacity match.",
    "benchmarks/sensitivity_results/logic_mamba2_dstate_32.json": "Logic baseline: Mamba-2 d_state=32.",
    "benchmarks/sensitivity_results/logic_mamba2_dstate_64.json": "Logic baseline: Mamba-2 d_state=64.",
    "benchmarks/sensitivity_results/logic_mamba2_dstate_128.json": "Logic baseline: Mamba-2 d_state=128 — widest state.",
}

RESULT_BLURBS = {
    ("tourism", "transformer"): "Tourism-AU Transformer baseline — sMAPE 171.47 (rank 4/5 on Tour.).",
    ("tourism", "informer"): "Tourism-AU Informer — sMAPE 163.42, second-best socio-economic forecaster here.",
    ("tourism", "mamba1"): "Tourism-AU Mamba-1 — sMAPE 156.44, underline (2nd) in Table 6 Tour. column.",
    ("tourism", "mamba2"): "Tourism-AU Mamba-2 — sMAPE 151.09, **best** Tour. sMAPE in committed runs.",
    ("tourism", "nlh_ssm"): "Tourism-AU NL-H-H-SSM — sMAPE 196.36; structure metric ACD 0.368 tied best.",
    ("labor", "transformer"): "Labour-AU Transformer — sMAPE 141.10, competitive but not best.",
    ("labor", "informer"): "Labour-AU Informer — sMAPE 176.35, weakest sMAPE among five models.",
    ("labor", "mamba1"): "Labour-AU Mamba-1 — sMAPE 139.83, underline second on Lab. column.",
    ("labor", "mamba2"): "Labour-AU Mamba-2 — sMAPE 53.60, **best** Lab. sMAPE by large margin.",
    ("labor", "nlh_ssm"): "Labour-AU NL-H-H-SSM — sMAPE 195.03; prioritizes ACD 0.368 over sMAPE here.",
    ("prison", "transformer"): "Prison-AU Transformer — sMAPE 165.17, mid-pack on Pris.",
    ("prison", "informer"): "Prison-AU Informer — sMAPE 155.80, underline second on Pris.",
    ("prison", "mamba1"): "Prison-AU Mamba-1 — sMAPE 172.61, worst sMAPE among five.",
    ("prison", "mamba2"): "Prison-AU Mamba-2 — sMAPE 113.23, **best** Pris. sMAPE.",
    ("prison", "nlh_ssm"): "Prison-AU NL-H-H-SSM — sMAPE 193.87; ACD 0.368 tied best expert metric.",
    ("m5", "transformer"): "M5-Walmart Transformer — RMSSE 5.40, underline **second**; strong industrial baseline.",
    ("m5", "informer"): "M5-Walmart Informer — RMSSE 7.05, weakest RMSSE on M5.",
    ("m5", "mamba1"): "M5-Walmart Mamba-1 — RMSSE 87.56, unstable scale on this hierarchy.",
    ("m5", "mamba2"): "M5-Walmart Mamba-2 — RMSSE 28.93, mid-tier among baselines.",
    ("m5", "nlh_ssm"): "M5-Walmart NL-H-H-SSM — RMSSE **4.30**, **best** M5 column in Table 6.",
    ("wiki", "transformer"): "Wiki-Traffic Transformer — RMSSE 1.12, tied NL-H on Wiki.",
    ("wiki", "informer"): "Wiki-Traffic Informer — RMSSE **0.60**, **best** Wiki column.",
    ("wiki", "mamba1"): "Wiki-Traffic Mamba-1 — RMSSE 0.90, third among five.",
    ("wiki", "mamba2"): "Wiki-Traffic Mamba-2 — RMSSE 0.85, underline second.",
    ("wiki", "nlh_ssm"): "Wiki-Traffic NL-H-H-SSM — RMSSE 1.12; ACD 0.368 tied best structure score.",
    ("electricity", "transformer"): "Electricity-L Transformer — RMSSE 2.39, mid industrial baseline.",
    ("electricity", "informer"): "Electricity-L Informer — RMSSE **0.74**, **best** Elec. column.",
    ("electricity", "mamba1"): "Electricity-L Mamba-1 — RMSSE 1.70, third place.",
    ("electricity", "mamba2"): "Electricity-L Mamba-2 — RMSSE 1.42, underline second.",
    ("electricity", "nlh_ssm"): "Electricity-L NL-H-H-SSM — RMSSE 2.33; flat ACD 0.368.",
    ("traffic", "transformer"): "Traffic-HTS Transformer — RMSSE 1.97, tied with NL-H on Traf.",
    ("traffic", "informer"): "Traffic-HTS Informer — RMSSE **0.47**, **best** Traf. column.",
    ("traffic", "mamba1"): "Traffic-HTS Mamba-1 — RMSSE 0.94, third.",
    ("traffic", "mamba2"): "Traffic-HTS Mamba-2 — RMSSE 0.70, underline second.",
    ("traffic", "nlh_ssm"): "Traffic-HTS NL-H-H-SSM — RMSSE 1.97; ACD 0.368 tied best.",
    ("solar", "transformer"): "Solar-HTS Transformer — RMSSE 8.92, tied NL-H worst tier.",
    ("solar", "informer"): "Solar-HTS Informer — RMSSE **0.46**, **best** Solr. column.",
    ("solar", "mamba1"): "Solar-HTS Mamba-1 — RMSSE 0.89, third.",
    ("solar", "mamba2"): "Solar-HTS Mamba-2 — RMSSE 0.76, underline second.",
    ("solar", "nlh_ssm"): "Solar-HTS NL-H-H-SSM — RMSSE 8.92; high sMAPE 199.5 on this stem.",
    ("logic", "transformer"): "LogicGraph Transformer — ACD 0.368 tied best; sMAPE 175.2.",
    ("logic", "informer"): "LogicGraph Informer — ACD 0.511, weaker structure fidelity.",
    ("logic", "mamba1"): "LogicGraph Mamba-1 — ACD **0.326**, **best** Logic. structure column.",
    ("logic", "mamba2"): "LogicGraph Mamba-2 — ACD 8.23, collapsed hierarchy embedding.",
    ("logic", "nlh_ssm"): "LogicGraph NL-H-H-SSM — ACD 0.368 tied best; saturates ablation sweeps.",
    ("medical", "transformer"): "Med-Diag-Path Transformer — ACD 0.368 tied best; sMAPE 131.0.",
    ("medical", "informer"): "Med-Diag-Path Informer — ACD 0.470, third on structure.",
    ("medical", "mamba1"): "Med-Diag-Path Mamba-1 — ACD **0.366**, **best** Med. ACD (underline).",
    ("medical", "mamba2"): "Med-Diag-Path Mamba-2 — ACD 9.58, poor cophenetic fit.",
    ("medical", "nlh_ssm"): "Med-Diag-Path NL-H-H-SSM — ACD 0.368 tied best; RMSSE 48.3 outlier.",
}


def _result_desc(path: str) -> str | None:
    name = Path(path).stem
    if name.endswith("_nlh_ssm"):
        stem, model = name[: -len("_nlh_ssm")], "nlh_ssm"
    else:
        stem, model = name.rsplit("_", 1)
    key = (stem, model)
    base = RESULT_BLURBS.get(key)
    if not base:
        return None
    fp = ROOT / path
    if fp.is_file():
        payload = json.loads(fp.read_text(encoding="utf-8"))
        hist = payload.get("history") or []
        m = payload.get("final_metrics") or (hist[-1] if hist else {})
        extras = []
        for k in ("smape", "rmsse", "acd"):
            v = m.get(k)
            if isinstance(v, (int, float)):
                extras.append(f"{k}={v:.4f}")
        if extras:
            return f"{base} Metrics: {', '.join(extras)}."
    return base


def _doc_stub_source(stub_path: str) -> str:
    """docs/descriptions/foo/bar.py.md -> foo/bar.py"""
    prefix = "docs/descriptions/"
    if not stub_path.startswith(prefix) or not stub_path.endswith(".md"):
        raise ValueError(stub_path)
    return stub_path[len(prefix) : -3]


def _collect_paths() -> list[str]:
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).strip().splitlines()
    paths = set(tracked)
    for rel in (
        "scripts/generate_repo_manifest.py",
        "scripts/upload_from_manifest.py",
        "scripts/commit_doc_layers.sh",
        "docs/README.md",
    ):
        if (ROOT / rel).is_file():
            paths.add(rel)
    desc_root = ROOT / "docs" / "descriptions"
    if desc_root.is_dir():
        for fp in desc_root.rglob("*.md"):
            paths.add(fp.relative_to(ROOT).as_posix())
    return sorted(paths)


def _describe_source(path: str, manifest_partial: dict[str, str]) -> str:
    if path in STATIC:
        return STATIC[path]
    if path in EXPERIMENT_CONFIGS:
        return EXPERIMENT_CONFIGS[path]
    if path in SENSITIVITY:
        return SENSITIVITY[path]
    if path.startswith("results/") and path.endswith(".json"):
        return _result_desc(path) or f"Table 6 run log: {path}"
    return manifest_partial.get(path) or f"Tracked project file: {path}"


def main() -> None:
    all_paths = _collect_paths()
    source_paths = [
        p
        for p in all_paths
        if not p.startswith("docs/descriptions/") and p != "docs/README.md"
    ]

    manifest: dict[str, str] = {}
    for path in source_paths:
        manifest[path] = _describe_source(path, manifest)

    if (ROOT / "docs/README.md").is_file():
        manifest["docs/README.md"] = STATIC["docs/README.md"]

    for path in all_paths:
        if not path.startswith("docs/descriptions/") or not path.endswith(".md"):
            continue
        src = _doc_stub_source(path)
        base = manifest.get(src) or _describe_source(src, manifest)
        manifest[path] = f"Review stub for `{src}` — {base}"

    out = ROOT / "REPO_MANIFEST.json"
    out.write_text(
        json.dumps({"files": manifest}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} entries to {out}")

    desc_root = ROOT / "docs" / "descriptions"
    for path, text in manifest.items():
        if path.startswith("docs/descriptions/") and path.endswith(".md"):
            continue
        if path in ("docs/README.md",):
            continue
        rel = Path(path)
        md_path = desc_root / rel.with_suffix(rel.suffix + ".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(
            f"# `{path}`\n\n{text}\n",
            encoding="utf-8",
        )
    print(f"Synced description stubs under docs/descriptions/")

    # results/README.md — 50 distinct rows (no shared template)
    res_lines = [
        "# results/",
        "",
        "Table 6 training logs: **50 JSON files**, each with a **unique** role and final metrics.",
        "",
        "See also [docs/descriptions/results/](../docs/descriptions/results/) for one stub per file.",
        "",
        "| File | Description |",
        "|------|-------------|",
    ]
    for path in sorted(k for k in manifest if k.startswith("results/") and k.endswith(".json")):
        res_lines.append(f"| `{path}` | {manifest[path]} |")
    res_lines.extend(
        [
            "",
            "Regenerate this table: `python scripts/generate_repo_manifest.py`",
            "",
        ]
    )
    (ROOT / "results" / "README.md").write_text("\n".join(res_lines), encoding="utf-8")
    print("Updated results/README.md")


if __name__ == "__main__":
    main()
