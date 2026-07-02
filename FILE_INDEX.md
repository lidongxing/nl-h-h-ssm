# File Index — NL-H-H-SSM

Complete catalog of every tracked file and directory in this repository.  
Paper: *Expert Systems with Applications* reproducibility bundle.

**Repository:** [https://github.com/lidongxing/nl-h-h-ssm](https://github.com/lidongxing/nl-h-h-ssm)

---

## Root (`/`)

| Path | Comment |
|------|---------|
| `README.md` | Main documentation: installation, dataset index, Table 6 / RQ2 / RQ3 reproduction commands |
| `FILE_INDEX.md` | This file — per-path purpose and paper linkage |
| `setup.py` | Setuptools entry; `pip install -e .` installs `nlh_ssm`, `csrc`, `models`, `experiments`, `benchmarks`; optional `[mamba]` extra pulls `mamba-ssm` + `triton` for Mamba baselines |
| `.gitignore` | Excludes `data/raw/*`, four large processed CSVs (>100 MB), `__pycache__`, `.egg-info`, generated `assets/*.png` |
| `.gitattributes` | Sets `*.sh text eol=lf` so bash scripts run correctly on Linux after clone from Windows |

---

## `nlh_ssm/` — core library

Hyperbolic hierarchical SSM implementation used by training and benchmarks.

| Path | Comment |
|------|---------|
| `nlh_ssm/__init__.py` | Package root; exposes version namespace for `import nlh_ssm` |
| `nlh_ssm/ops/hyperbolic.py` | Poincaré ball math: exponential / logarithmic maps, Möbius addition, hyperbolic distance, projection; foundation for PH-Scan and NL-H blocks |
| `nlh_ssm/ops/check_nan.py` | Runtime NaN/Inf detection hooks for unstable hyperbolic gradients during training |
| `nlh_ssm/modules/nlh_block.py` | **Main layer**: stacks hyperbolic state update, PH-Scan recurrence, ACG gating; supports ablation flags `wo_hyp`, `wo_acg`, `wo_ph_scan` (RQ3) |
| `nlh_ssm/modules/hierarchy_embedding.py` | Embeds hierarchical series IDs (slash-delimited paths) into curvature-aware features for ACG |
| `nlh_ssm/models/mixer_seq_simple.py` | Full sequence model: input projection → stacked NL-H blocks → readout head; used by `run_exp.py` and `speed_test.py` |
| `nlh_ssm/models/selective_state_update.py` | Mamba-compatible selective SSM update interface bridged to hyperbolic state dynamics |
| `nlh_ssm/data/__init__.py` | Re-exports `get_dataloader` from loader |
| `nlh_ssm/data/loader.py` | Reads long-format CSV (`id`, `time`, `value`); builds sliding windows; infers hierarchy depth from ID delimiter for ACD / ACG |
| `nlh_ssm/metrics/__init__.py` | Metric submodule exports |
| `nlh_ssm/metrics/accuracy.py` | **Forecast metrics**: sMAPE (socio-economic), RMSSE (industrial/IoT), TW-MSE, CRPS — used in Table 6 columns |
| `nlh_ssm/metrics/distance.py` | **Structure metrics**: ACD (Average Cophenetic Distance), MAT — Logic/Medical Table 6 columns |
| `nlh_ssm/utils/__init__.py` | Utility exports |
| `nlh_ssm/utils/metrics.py` | Low-level metric kernels shared by `accuracy.py` and `distance.py` |

---

## `csrc/` — PH-Scan kernel

| Path | Comment |
|------|---------|
| `csrc/ph_scan_kernel.py` | Parallel prefix scan on the Poincaré manifold (PH-Scan); Triton JIT when `NLH_SSM_USE_TRITON≠0`, else PyTorch fallback; kernel-level contribution in paper §4 |
| `csrc/cauchy.py` | Cauchy / rational function helpers for stable hyperbolic scan accumulation |

---

## `models/` — thin facades for experiments

| Path | Comment |
|------|---------|
| `models/__init__.py` | Registers model names consumed by `experiments/run_exp.py` |
| `models/nlh_ssm.py` | Wraps `MixerSeqSimple` with paper hyperparameters (`nlh_c_base`, `nlh_num_layers`, `nlh_expand`) |
| `models/ph_scan.py` | Standalone PH-Scan module facade for ablation and unit tests |
| `models/acg_gate.py` | Adaptive Curvature Gate (ACG): depth-dependent curvature scaling from hierarchy metadata |

---

## `experiments/` — training entry points

| Path | Comment |
|------|---------|
| `experiments/__init__.py` | Package marker for setuptools |
| `experiments/.gitkeep` | Keeps empty directory in git before configs were added |
| `experiments/run_exp.py` | **Primary CLI** — trains `transformer` / `informer` / `mamba1` / `mamba2` / `nlh_ssm` on one CSV; writes `results/{stem}_{model}.json` with metrics + history; loads `configs/nlh_tuned.yaml` via `--nlh_hparams_file` |
| `experiments/run_main.py` | YAML-config wrapper around `run_exp.py` for single-dataset runs |
| `experiments/launch_device.py` | Parses `device` from YAML: `cpu`, `cuda:0`, multi-GPU DDP `"cuda:0,1"` |
| `experiments/train_all.sh` | Loops over `experiments/configs/*.yaml` and launches training |
| `experiments/configs/tourism.yaml` | Example run config: Tourism-AU CSV path, epochs, optional device |
| `experiments/configs/labor.yaml` | Example run config: Labour-AU |
| `experiments/configs/prison.yaml` | Example run config: Prison-AU |
| `experiments/configs/m5.yaml` | Example run config: M5-Walmart (large CSV; path only) |
| `experiments/configs/wiki.yaml` | Example run config: Wiki-Traffic |
| `experiments/configs/electricity.yaml` | Example run config: Electricity-L |
| `experiments/configs/traffic.yaml` | Example run config: Traffic-HTS |
| `experiments/configs/solar.yaml` | Example run config: Solar-HTS |
| `experiments/configs/logic.yaml` | Example run config: LogicGraph (ACD metric) |
| `experiments/configs/medical.yaml` | Example run config: Med-Diag-Path (ACD metric) |

---

## `configs/` — Table 6 NL-H protocol

| Path | Comment |
|------|---------|
| `configs/nlh_tuned.yaml` | **Paper Table 6 protocol**: per-dataset `seq_len`, `stride`, `nlh_lr`, `grad_clip_norm`, `nlh_c_base`, `nlh_num_layers`, `nlh_expand`; shared by `run_table6_all.sh` for fair window sizes |
| `configs/nlh_tuned.example.yaml` | Blank template showing YAML schema for `tune_nlh_ssm.py` output |

---

## `data/` — acquisition and preprocessing

| Path | Comment |
|------|---------|
| `data/.gitkeep` | Placeholder for data root |
| `data/download_all.sh` | Downloads ABS (tourism, labour, prison), Zenodo Monash (wiki, electricity, traffic, solar); documents Kaggle M5 manual step |
| `data/preprocess.py` | Converts raw zips/xlsx/json → unified long CSV `{id, time, value}`; supports `--dataset all` and subsampling flags for smoke tests |
| `data/raw/.gitkeep` | Raw downloads live here locally; **not committed** (gitignored except this marker) |
| `data/processed/.gitkeep` | Output directory marker |
| `data/processed/tourism.csv` | Tourism-AU processed series (~0.3 MB); paper column Tour. / sMAPE |
| `data/processed/labor.csv` | Labour-AU processed series (~0.8 MB); paper column Lab. |
| `data/processed/prison.csv` | Prison-AU processed series; paper column Pris. |
| `data/processed/wiki.csv` | Wiki-Traffic processed series (~6.5 MB); paper column Wiki / RMSSE |
| `data/processed/logic.csv` | LogicGraph hierarchical logic traces (~0.9 MB); paper column Logic. / ACD |
| `data/processed/medical.csv` | Med-Diag-Path diagnostic paths (~4.5 MB); paper column Med. / ACD |
| `data/processed/m5.csv` | M5-Walmart — **gitignored** (216 MB); regenerate with `preprocess.py` |
| `data/processed/electricity.csv` | Electricity-L — **gitignored** (141 MB) |
| `data/processed/traffic.csv` | Traffic-HTS — **gitignored** (261 MB) |
| `data/processed/solar.csv` | Solar-HTS — **gitignored** (106 MB) |

---

## `scripts/` — orchestration and paper tables

| Path | Comment |
|------|---------|
| `scripts/run_table6_all.sh` | **Table 6 full campaign**: 10 stems × 5 models, 40 epochs, windows from `nlh_tuned.yaml`; subcommands `nlh_only`, `baselines_only`, `rank`, `tex` |
| `scripts/run_table6_fair_baselines.sh` | Re-trains baselines only with same `seq_len`/`stride` as NL-H yaml (no `nlh_lr`) |
| `scripts/table6_nlh_campaign.sh` | Optional two-phase NL-H workflow: `tune_*` then `train_*` per dataset subset |
| `scripts/generate_table_6.py` | Reads `results/*.json` → LaTeX Table 6 (`--format harv` for elsarticle two-row header; no Impv. column) |
| `scripts/rank_table6.py` | Computes per-column ranks and average rank across 10 datasets |
| `scripts/validate_table3_stats.py` | Validates dataset layer count and branching factor against paper Table 3 expected values |
| `scripts/tune_nlh_ssm.py` | Grid search NL-H hyperparameters on validation windows; writes/updates `configs/nlh_tuned.yaml` |
| `scripts/run_speed_forward_bench.sh` | Wrapper: `python benchmarks/speed_test.py --forward-only` → Figure 7 data |
| `scripts/run_speed_training_bench.sh` | Wrapper: full fwd+bwd training-step benchmark → Table 7 / Fig. scalability |
| `scripts/run_rq3_ablation.sh` | Runs component ablation (`wo_hyp`, `wo_acg`, `wo_ph_scan`) on Logic + M5 |
| `scripts/run_parameter_sensitivity.sh` | Sweeps `nlh_c_base`, `nlh_expand`, Mamba-2 `d_state` on Logic |
| `scripts/run_rq3_enhancements_all.sh` | Chains ablation → sensitivity → training speed (RQ3 pipeline) |
| `scripts/aggregate_component_ablation.py` | Merges per-run ablation JSON into `benchmarks/ablation_model_components.json` |
| `scripts/aggregate_parameter_sensitivity.py` | Merges sensitivity runs into `benchmarks/figure10_parameter_sensitivity.json` |

---

## `benchmarks/` — speed, ablation, figures

| Path | Comment |
|------|---------|
| `benchmarks/__init__.py` | Package marker |
| `benchmarks/speed_test.py` | GPU throughput driver: NL-H-H-SSM vs Mamba-2 vs Transformer at L=1k…128k; writes JSON + optional Figure 7 PNG |
| `benchmarks/speed_results/speed_bench.json` | **Committed** forward-only results (NVIDIA A800, CentOS); paper §6.2 / Figure 7 |
| `benchmarks/speed_results/speed_bench_train.json` | **Committed** forward+backward training-step throughput; paper Table 7 |
| `benchmarks/speed_results/speed_bench_synthetic.json` | Sanity-check runs on synthetic lengths (debug / smoke) |
| `benchmarks/ablation_model_components.json` | **Committed** RQ3 component ablation summary (Hyp / PH-Scan / ACG removals); Figure 9 source |
| `benchmarks/ablation_operator.json` | Operator-level ablation (Euclidean vs hyperbolic scan variants) |
| `benchmarks/figure10_parameter_sensitivity.json` | **Committed** aggregated Logic sensitivity curves; Figure 10 source |
| `benchmarks/sensitivity_results/logic_cbase_0.1.json` | Raw run: Logic dataset, `nlh_c_base=0.1` |
| `benchmarks/sensitivity_results/logic_cbase_0.5.json` | Raw run: `nlh_c_base=0.5` |
| `benchmarks/sensitivity_results/logic_cbase_1.0.json` | Raw run: `nlh_c_base=1.0` |
| `benchmarks/sensitivity_results/logic_cbase_1.5.json` | Raw run: `nlh_c_base=1.5` |
| `benchmarks/sensitivity_results/logic_cbase_2.0.json` | Raw run: `nlh_c_base=2.0` |
| `benchmarks/sensitivity_results/logic_expand_1.json` | Raw run: `nlh_expand=1` (width multiplier) |
| `benchmarks/sensitivity_results/logic_expand_2.json` | Raw run: `nlh_expand=2` |
| `benchmarks/sensitivity_results/logic_expand_4.json` | Raw run: `nlh_expand=4` |
| `benchmarks/sensitivity_results/logic_expand_8.json` | Raw run: `nlh_expand=8` |
| `benchmarks/sensitivity_results/logic_mamba2_dstate_16.json` | Raw run: Mamba-2 baseline with `d_state=16` on Logic |
| `benchmarks/sensitivity_results/logic_mamba2_dstate_32.json` | Raw run: `d_state=32` |
| `benchmarks/sensitivity_results/logic_mamba2_dstate_64.json` | Raw run: `d_state=64` |
| `benchmarks/sensitivity_results/logic_mamba2_dstate_128.json` | Raw run: `d_state=128` |
| `benchmarks/model_ablation_figure9.py` | Plots component ablation bar chart → `assets/figure9_component_ablation.png` |
| `benchmarks/parameter_sensitivity_figure10.py` | Plots sensitivity curves → `assets/figure10_parameter_sensitivity.png` |
| `benchmarks/operator_ablation.py` | Runner for operator-level ablation experiments |
| `benchmarks/generate_table.py` | Legacy LaTeX generator for six-dataset Table 5 |
| `benchmarks/plot_radar.py` | Multi-metric radar chart (illustrative comparison figure) |
| `benchmarks/plot_curvature_heatmap.py` | Regime / curvature heatmap schematic for paper figures |
| `benchmarks/plot_poincare_vs_euclidean.py` | Poincaré vs Euclidean embedding schematic |
| `benchmarks/figure_architecture_english.py` | Generates English architecture diagram for paper |

---

## `results/` — Table 6 experiment logs (50 files)

Each JSON contains: `dataset`, `model`, `hyperparams`, per-epoch `history`, and final `metrics` (sMAPE, RMSSE, and/or ACD depending on dataset group). Produced by `experiments/run_exp.py` under the Table 6 protocol (`EPOCHS=40`, shared windows).

| Path | Comment |
|------|---------|
| `results/tourism_{model}.json` | Tourism-AU; models: transformer, informer, mamba1, mamba2, nlh_ssm |
| `results/labor_{model}.json` | Labour-AU |
| `results/prison_{model}.json` | Prison-AU |
| `results/m5_{model}.json` | M5-Walmart (RMSSE); NL-H best RMSSE column in Table 6 |
| `results/wiki_{model}.json` | Wiki-Traffic |
| `results/electricity_{model}.json` | Electricity-L |
| `results/traffic_{model}.json` | Traffic-HTS |
| `results/solar_{model}.json` | Solar-HTS |
| `results/logic_{model}.json` | LogicGraph (ACD) |
| `results/medical_{model}.json` | Med-Diag-Path (ACD) |

Replace `{model}` with: `transformer`, `informer`, `mamba1`, `mamba2`, `nlh_ssm`.

---

## `tests/` — unit tests

| Path | Comment |
|------|---------|
| `tests/test_hyperbolic.py` | Tests exp/log map round-trip and Möbius addition on Poincaré ball |
| `tests/test_ph_scan.py` | Smoke test PH-Scan kernel output shape and finiteness |
| `tests/test_metrics.py` | Tests sMAPE, RMSSE, ACD against hand-computed examples |

Run: `pytest tests/` from repo root.

---

## `assets/` — generated figures

| Path | Comment |
|------|---------|
| `assets/.gitkeep` | Placeholder; PNG/PDF outputs are gitignored and rebuilt by benchmark scripts |

Expected outputs (local only):

| Generated file | Producer |
|----------------|----------|
| `assets/figure7_speed_scaling.png` | `run_speed_forward_bench.sh` |
| `assets/figure9_component_ablation.png` | `model_ablation_figure9.py` |
| `assets/figure10_parameter_sensitivity.png` | `parameter_sensitivity_figure10.py` |

---

## Quick map: paper artifact → file

| Paper artifact | Primary file(s) |
|----------------|-----------------|
| Table 6 main results | `results/*_*.json` → `scripts/generate_table_6.py` |
| Table 3 dataset stats | `data/preprocess.py` → `scripts/validate_table3_stats.py` |
| Table 7 training throughput | `benchmarks/speed_results/speed_bench_train.json` |
| Figure 7 forward scaling | `benchmarks/speed_results/speed_bench.json` |
| Figure 9 component ablation | `benchmarks/ablation_model_components.json` |
| Figure 10 sensitivity | `benchmarks/figure10_parameter_sensitivity.json` |
| NL-H hyperparameters | `configs/nlh_tuned.yaml` |
| Fair comparison protocol | `scripts/run_table6_all.sh` |
