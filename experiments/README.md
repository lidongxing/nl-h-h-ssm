# experiments/

Training drivers and per-dataset YAML examples.

| File | Role |
|------|------|
| `run_exp.py` | **Main CLI** — one dataset × one model → `results/{stem}_{model}.json` |
| `run_main.py` | YAML launcher |
| `launch_device.py` | GPU / DDP device parsing |
| `train_all.sh` | Batch train all configs |
| `configs/*.yaml` | Example paths and epochs for each Table 6 dataset |

See [FILE_INDEX.md](../FILE_INDEX.md#experiments--training-entry-points).
