# scripts/

Reproduction orchestration for Table 6, LaTeX export, tuning, and RQ3.

| Script | Role |
|--------|------|
| `run_table6_all.sh` | Full Table 6: 10 × 5 models, 40 epochs |
| `generate_table_6.py` | JSON → LaTeX Table 6 |
| `rank_table6.py` | Average ranks |
| `run_rq3_enhancements_all.sh` | Ablation + sensitivity + speed |
| `tune_nlh_ssm.py` | Hyperparameter grid search |

See [FILE_INDEX.md](../FILE_INDEX.md#scripts--orchestration-and-paper-tables).
