# data/

Dataset download, preprocessing, and processed CSVs.

| Path | Role |
|------|------|
| `download_all.sh` | Fetch public raw files into `raw/` |
| `preprocess.py` | Build unified `{id, time, value}` CSVs in `processed/` |
| `raw/` | Raw downloads (gitignored locally) |
| `processed/` | Six small CSVs committed; four large CSVs rebuilt locally |

```bash
bash data/download_all.sh
python data/preprocess.py --dataset all --raw-root data/raw --out-dir data/processed
```

See [FILE_INDEX.md](../FILE_INDEX.md#data--acquisition-and-preprocessing).
