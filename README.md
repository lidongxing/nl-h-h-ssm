# NL-H-H-SSM

Official implementation of **NL-H-H-SSM** (Non-Linear Hyperbolic Hierarchical State Space Model): a hyperbolic SSM with PH-Scan and ACG for hierarchical time-series forecasting and structural fidelity (ACD).

This repository supports reproducibility of the experiments reported in our *Expert Systems with Applications* submission: **Table 6** (ten datasets), **speed benchmarks** (forward / training throughput), and **RQ3 ablations** (component removal and parameter sensitivity).

**Repository:** [https://github.com/lidongxing/nl-h-h-ssm](https://github.com/lidongxing/nl-h-h-ssm)

**Per-file descriptions (unique, not duplicated):**

| Resource | Format |
|----------|--------|
| [REPO_MANIFEST.json](REPO_MANIFEST.json) | JSON map: every tracked path → one specific sentence |
| [FILE_INDEX.md](FILE_INDEX.md) | Human-readable catalog grouped by folder |
| [docs/descriptions/](docs/descriptions/) | One markdown stub per source file (mirrors repo tree) |

> GitHub’s **“Last commit message”** column on the file list reflects the **last git commit that touched that file**, not a custom label.  
> Code and JSON logs may still show an older shared message; use **`docs/descriptions/<path>.md`** or **`REPO_MANIFEST.json`** for the dedicated explanation of each path.

Each major folder also has its own `README.md` (e.g. `nlh_ssm/README.md`, `results/README.md`).

---

## Repository layout (overview)

| Path | Purpose |
|------|---------|
| `nlh_ssm/` | Core library: hyperbolic ops, NL-H blocks, data loaders, metrics |
| `csrc/` | PH-Scan kernel (`ph_scan_kernel.py`) |
| `models/` | Thin model facades used by `run_exp.py` |
| `experiments/` | Training entry points (`run_exp.py`, `run_main.py`) |
| `configs/` | NL-H hyperparameter YAML used in Table 6 |
| `data/` | Download script, preprocessing, processed CSVs |
| `scripts/` | Table 6 campaign, LaTeX export, tuning, validation |
| `benchmarks/` | Speed tests, ablation / sensitivity, figure scripts |
| `results/` | **Committed** Table 6 metrics (`{stem}_{model}.json`) |
| `tests/` | Unit tests |
| `assets/` | Generated figures (PNG/PDF; not committed by default) |

---

## File and directory reference

> **Complete per-file comments:** see **[FILE_INDEX.md](FILE_INDEX.md)** (150 tracked paths).  
> **Per-folder summaries:** see `README.md` inside each directory below.

Brief overview of top-level paths:

### Root files

| File | Description |
|------|-------------|
| `README.md` | This document: install, data pipeline, reproduction commands |
| `setup.py` | Package definition (`pip install -e .`); optional `[mamba]` extra for baselines |
| `.gitignore` | Excludes raw data, large CSVs (>100 MB), caches, generated assets |
| `.gitattributes` | Forces LF line endings for shell scripts |

### `nlh_ssm/` — core Python package

| Path | Description |
|------|-------------|
| `__init__.py` | Package marker |
| `ops/hyperbolic.py` | Poincaré ball operations (exp/log map, Möbius add, distance) |
| `ops/check_nan.py` | NaN guards for hyperbolic numerics |
| `modules/nlh_block.py` | NL-H block: hyperbolic SSM layer with PH-Scan and ACG |
| `modules/hierarchy_embedding.py` | Hierarchy-aware positional / structural embedding |
| `models/mixer_seq_simple.py` | Sequence mixer backbone wiring NL-H blocks |
| `models/selective_state_update.py` | Selective state update (Mamba-style interface) |
| `data/loader.py` | Sliding-window CSV loader; hierarchy metadata from series IDs |
| `metrics/accuracy.py` | sMAPE, RMSSE, TW-MSE, CRPS |
| `metrics/distance.py` | ACD (Average Cophenetic Distance), MAT |
| `utils/metrics.py` | Shared metric primitives |

### `csrc/` — kernel extensions

| Path | Description |
|------|-------------|
| `ph_scan_kernel.py` | PH-Scan parallel prefix scan on the Poincaré manifold (Triton / fallback) |
| `cauchy.py` | Cauchy / rational-kernel helper used by scan implementation |

### `models/` — experiment facades

| Path | Description |
|------|-------------|
| `nlh_ssm.py` | High-level NL-H-H-SSM model wrapper for `run_exp.py` |
| `ph_scan.py` | PH-Scan module facade |
| `acg_gate.py` | Adaptive Curvature Gate (ACG) |
| `__init__.py` | Re-exports model entry points |

### `experiments/` — training drivers

| Path | Description |
|------|-------------|
| `run_exp.py` | **Main CLI**: train any model on a CSV; writes `results/{stem}_{model}.json` |
| `run_main.py` | YAML-driven training launcher |
| `launch_device.py` | Parses `device` field (single GPU, multi-GPU DDP) |
| `train_all.sh` | Batch train from `experiments/configs/*.yaml` |
| `configs/*.yaml` | Per-dataset example configs (paths, epochs, optional device) |

### `configs/` — Table 6 NL-H hyperparameters

| Path | Description |
|------|-------------|
| `nlh_tuned.yaml` | **Paper protocol**: per-dataset `seq_len`, `stride`, LR, `c_base`, layers |
| `nlh_tuned.example.yaml` | Template for tuning output |

### `data/` — datasets

| Path | Description |
|------|-------------|
| `download_all.sh` | Download ABS, Zenodo Monash, and other public sources into `data/raw/` |
| `preprocess.py` | Convert raw files to unified long-format `{stem}.csv` in `data/processed/` |
| `raw/.gitkeep` | Placeholder; raw downloads are gitignored |
| `processed/.gitkeep` | Placeholder for processed output |
| `processed/tourism.csv` | Tourism-AU (committed, small) |
| `processed/labor.csv` | Labour-AU (committed) |
| `processed/prison.csv` | Prison-AU (committed) |
| `processed/wiki.csv` | Wiki-Traffic (committed) |
| `processed/logic.csv` | LogicGraph expert-system series (committed) |
| `processed/medical.csv` | Med-Diag-Path (committed) |
| `processed/m5.csv` | M5-Walmart — **gitignored** (>100 MB); rebuild locally |
| `processed/electricity.csv` | Electricity-L — **gitignored** |
| `processed/traffic.csv` | Traffic-HTS — **gitignored** |
| `processed/solar.csv` | Solar-HTS — **gitignored** |

### `scripts/` — reproduction orchestration

| Path | Description |
|------|-------------|
| `run_table6_all.sh` | Full Table 6: 10 datasets × 5 models, 40 epochs |
| `run_table6_fair_baselines.sh` | Re-run baselines with NL-H window sizes from YAML |
| `table6_nlh_campaign.sh` | Optional tune-then-train NL-H campaign per dataset |
| `generate_table_6.py` | Build Table 6 LaTeX from `results/*.json` |
| `rank_table6.py` | Print average ranks per model |
| `validate_table3_stats.py` | Check dataset layer/branching stats vs paper Table 3 |
| `tune_nlh_ssm.py` | Grid search NL-H hyperparameters on validation windows |
| `run_speed_forward_bench.sh` | Forward throughput → `speed_bench.json` + Figure 7 |
| `run_speed_training_bench.sh` | Training-step throughput → `speed_bench_train.json` |
| `run_rq3_ablation.sh` | Component ablation runs (Logic + M5) |
| `run_parameter_sensitivity.sh` | Logic sensitivity sweeps |
| `run_rq3_enhancements_all.sh` | Runs all RQ3 steps in order |
| `aggregate_component_ablation.py` | Merge ablation JSON → `ablation_model_components.json` |
| `aggregate_parameter_sensitivity.py` | Merge sensitivity JSON → `figure10_parameter_sensitivity.json` |

### `benchmarks/` — benchmarks and figures

| Path | Description |
|------|-------------|
| `speed_test.py` | Throughput / VRAM driver for forward and training steps |
| `speed_results/speed_bench.json` | **Committed** forward-only results (A800) |
| `speed_results/speed_bench_train.json` | **Committed** fwd+bwd training throughput |
| `speed_results/speed_bench_synthetic.json` | Synthetic-length sanity checks |
| `ablation_model_components.json` | **Committed** RQ3 component ablation summary |
| `figure10_parameter_sensitivity.json` | **Committed** parameter sensitivity summary |
| `sensitivity_results/*.json` | Per-run sensitivity outputs (Logic sweeps) |
| `ablation_operator.json` | Operator-level ablation summary |
| `model_ablation_figure9.py` | Plot component ablation (Figure 9) |
| `parameter_sensitivity_figure10.py` | Plot sensitivity curves (Figure 10) |
| `operator_ablation.py` | Operator ablation runner |
| `generate_table.py` | Legacy Table 5 LaTeX from six-dataset results |
| `plot_radar.py` | Radar chart for multi-metric comparison |
| `plot_curvature_heatmap.py` | Curvature / regime heatmap |
| `plot_poincare_vs_euclidean.py` | Poincaré vs Euclidean schematic |
| `figure_architecture_english.py` | Architecture diagram generator |

### `results/` — Table 6 experiment logs

50 JSON files: `{stem}_{model}.json` for `stem ∈ {tourism,labor,prison,m5,wiki,electricity,traffic,solar,logic,medical}` and `model ∈ {transformer,informer,mamba1,mamba2,nlh_ssm}`. Each file stores final metrics, hyperparameters, and training history.

### `tests/` — unit tests

| Path | Description |
|------|-------------|
| `test_hyperbolic.py` | Hyperbolic op correctness |
| `test_ph_scan.py` | PH-Scan kernel smoke tests |
| `test_metrics.py` | Metric computation tests |

### `assets/` — generated figures

| Path | Description |
|------|-------------|
| `.gitkeep` | Placeholder; PNG/PDF outputs from benchmark scripts (gitignored) |

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 with CUDA (recommended for full reproduction)
- Linux or WSL recommended for shell scripts (`.sh` files use LF line endings)

Optional baseline dependencies:

```bash
pip install -e ".[mamba]"   # Mamba-1 / Mamba-2 baselines (mamba-ssm + triton)
```

Core install:

```bash
cd NL-H-H-SSM
pip install -e .
pytest tests/
```

Environment variables (optional):

| Variable | Effect |
|----------|--------|
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `NLH_SSM_USE_TRITON=0` | Disable Triton path in PH-Scan (CPU / compatibility fallback) |

---

## What is committed vs. rebuilt locally

| Artifact | In repo | Notes |
|----------|---------|-------|
| `results/*.json` | Yes | Paper Table 6 numbers (50 runs: 10 datasets × 5 models) |
| `benchmarks/speed_results/*.json` | Yes | Forward / training throughput |
| `benchmarks/ablation_*.json`, `sensitivity_results/` | Yes | RQ3 ablation & sensitivity |
| `data/processed/*.csv` (small) | Partial | tourism, labor, prison, logic, medical, wiki (< 100 MB each) |
| `data/processed/{m5,traffic,electricity,solar}.csv` | **No** | Exceed GitHub 100 MB limit — regenerate locally |
| `data/raw/` | **No** | Download with `data/download_all.sh` |
| `assets/*.png` | **No** | Regenerate via benchmark scripts |

---

## Dataset index (Table 6)

| Paper name | CSV stem | Group | Primary metric |
|------------|----------|-------|----------------|
| Tourism-AU | `tourism` | Socio-Economic | sMAPE ↓ |
| Labour-AU | `labor` | Socio-Economic | sMAPE ↓ |
| Prison-AU | `prison` | Socio-Economic | sMAPE ↓ |
| M5-Walmart | `m5` | Industrial & IoT | RMSSE ↓ |
| Wiki-Traffic | `wiki` | Industrial & IoT | RMSSE ↓ |
| Electricity-L | `electricity` | Industrial & IoT | RMSSE ↓ |
| Traffic-HTS | `traffic` | Industrial & IoT | RMSSE ↓ |
| Solar-HTS | `solar` | Industrial & IoT | RMSSE ↓ |
| LogicGraph | `logic` | Expert Systems | ACD ↓ |
| Med-Diag-Path | `medical` | Expert Systems | ACD ↓ |

Window sizes (`seq_len`, `stride`) and NL-H hyperparameters are defined per dataset in `configs/nlh_tuned.yaml`.

---

## Data acquisition and preprocessing

### 1. Download raw files

```bash
bash data/download_all.sh
```

Sources include ABS (tourism, labour, prison), Zenodo Monash bundles (wiki, electricity, traffic, solar), Kaggle M5 (manual — see script comments), and LogicGraph / medical paths documented in `data/preprocess.py`.

On Windows without bash, run the `curl` URLs from `data/download_all.sh` manually into `data/raw/`.

### 2. Preprocess to unified CSV format

```bash
python data/preprocess.py --dataset all --raw-root data/raw --out-dir data/processed
```

Optional subsampling for quick smoke tests:

```bash
python data/preprocess.py --dataset all --raw-root data/raw --out-dir data/processed \
  --max-series 2500 --m5-max-rows 6000
```

### 3. Validate dataset statistics (Table 3)

After preprocessing, export or build a summary CSV/JSON and run:

```bash
python scripts/validate_table3_stats.py --input path/to/table3_summary.csv
```

---

## Reproducing Table 6 (main results)

**Protocol** (`scripts/run_table6_all.sh`):

- 10 datasets × 5 models: `transformer`, `informer`, `mamba1`, `mamba2`, `nlh_ssm`
- **40 epochs** (override with `EPOCHS=…`)
- Shared windows from `configs/nlh_tuned.yaml` for all models
- Baselines: `lr=0.001`; NL-H uses per-dataset fields from the same YAML
- Output: `results/{stem}_{model}.json`

### Verify committed results (no GPU)

```bash
python scripts/rank_table6.py --results_dir results
python scripts/generate_table_6.py --results_dir results --format harv \
  --out table6_harv_fragment.tex
```

Paste `table6_harv_fragment.tex` into the paper source (`tab:main_results`).

### Full re-run (GPU, multi-day)

```bash
export CUDA_VISIBLE_DEVICES=0
EPOCHS=40 bash scripts/run_table6_all.sh
```

Subcommands:

```bash
EPOCHS=40 bash scripts/run_table6_all.sh nlh_only        # NL-H only
EPOCHS=40 bash scripts/run_table6_all.sh baselines_only    # four baselines
bash scripts/run_table6_all.sh rank                        # average ranks
bash scripts/run_table6_all.sh tex                         # LaTeX fragment
```

Single-dataset smoke test:

```bash
python experiments/run_exp.py --model nlh_ssm \
  --dataset data/processed/logic.csv \
  --nlh_hparams_file configs/nlh_tuned.yaml --epochs 5
```

---

## Speed benchmarks (RQ2)

Forward-only throughput (Figure 7):

```bash
bash scripts/run_speed_forward_bench.sh
# → benchmarks/speed_results/speed_bench.json
# → assets/figure7_speed_scaling.png
```

Training-step throughput (forward + backward):

```bash
bash scripts/run_speed_training_bench.sh
# → benchmarks/speed_results/speed_bench_train.json
```

Committed JSON files in `benchmarks/speed_results/` match the hardware notes in the paper (NVIDIA A800, CentOS).

---

## Ablations and sensitivity (RQ3)

```bash
EPOCHS=40 bash scripts/run_rq3_enhancements_all.sh
```

This runs, in order:

1. Component ablation (`wo_hyp`, `wo_acg`, `wo_ph_scan`) on Logic + M5  
2. Parameter sensitivity on Logic (`c_base`, expand, Mamba-2 `d_state`)  
3. Training-step speed benchmark  

Aggregated JSON and figures:

- `benchmarks/ablation_model_components.json` → `assets/figure9_component_ablation.png`
- `benchmarks/figure10_parameter_sensitivity.json` → `assets/figure10_parameter_sensitivity.png`

---

## Hyperparameter tuning (optional)

Grid search on a validation subsample, then write `configs/nlh_tuned.yaml`:

```bash
python scripts/tune_nlh_ssm.py --data_dir data/processed --tune_epochs 5 --max_windows 8000
# quick pass:
python scripts/tune_nlh_ssm.py --data_dir data/processed --quick --tune_epochs 3 --max_windows 4000
```

Template: `configs/nlh_tuned.example.yaml`.

---

## Models compared

| Model | Implementation |
|-------|----------------|
| Transformer | `experiments/run_exp.py` |
| Informer | `experiments/run_exp.py` |
| Mamba-1 / Mamba-2 | `mamba-ssm` (optional extra) |
| **NL-H-H-SSM** | `nlh_ssm` + PH-Scan + ACG |

HiM / HiSS variants cited in the paper taxonomy are **not** included (no public reference implementation at experiment time).

---

## Citation

If you use this code, please cite our paper (bibtex to be added upon publication):

```bibtex
@article{nlhhssm2026,
  title   = {Non-linear Hyperbolic State Space Models via M{\"o}bius Algebra for Hierarchical Time Series Forecasting},
  journal = {Expert Systems with Applications},
  year    = {2026}
}
```

---

## License

Research code released for reproducibility. Specify license before public release if required by your institution.
