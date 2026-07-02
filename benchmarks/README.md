# benchmarks/

Speed tests, ablation/sensitivity JSON, and figure generators.

| Path | Role |
|------|------|
| `speed_test.py` | GPU throughput driver |
| `speed_results/*.json` | Committed A800 forward / training benchmarks |
| `ablation_model_components.json` | RQ3 component ablation (Figure 9) |
| `figure10_parameter_sensitivity.json` | RQ3 sensitivity (Figure 10) |
| `sensitivity_results/` | Per-sweep raw JSON on Logic |
| `*_figure*.py`, `plot_*.py` | Figure scripts → `assets/` |

See [FILE_INDEX.md](../FILE_INDEX.md#benchmarks--speed-ablation-figures).
