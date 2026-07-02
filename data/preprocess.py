"""
Build unified long-format CSVs for Table-3 / Table-6 datasets (id, time, value).

Outputs match ``nlh_ssm.data.loader.get_dataloader(..., schema="long")`` defaults:
  - id_col:   "id"
  - time_col: "time"
  - value_cols: ("value",)

Hierarchy depth for ACG is inferred from ``id`` using ``/`` segments (see loader).

Usage (from repo root ``NL-H-H-SSM``)::

  python data/preprocess.py --dataset all --raw-root data/raw --out-dir data/processed
  python data/preprocess.py --dataset m5 --raw-root data/raw --out-dir data/processed

Requires: ``pip install pandas``; for .xlsx also ``pip install openpyxl``.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(s: str, max_len: int = 80) -> str:
    t = re.sub(r"\W+", "_", str(s).strip()).strip("_")
    return t[:max_len] if t else "na"


def _ensure_out(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _first_match(root: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pat in patterns:
        p = root / pat
        if p.is_file() or p.is_dir():
            return p
        if "*" in pat or "?" in pat:
            hits = sorted(root.glob(pat))
            for h in hits:
                if h.is_file() or h.is_dir():
                    return h
    return None


def _wide_first_col_to_long(df: pd.DataFrame, *, id_col: Optional[str] = None) -> pd.DataFrame:
    """Monash-style wide: first column = series id, remaining = time steps."""
    if df.empty:
        raise ValueError("empty dataframe")
    ic = id_col or str(df.columns[0])
    if ic not in df.columns:
        ic = str(df.columns[0])
    value_cols = [c for c in df.columns if c != ic]
    if not value_cols:
        raise ValueError("no time columns after id column")
    long_df = df.melt(id_vars=[ic], var_name="time", value_name="value")
    long_df.rename(columns={ic: "id"}, inplace=True)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])
    tnum = pd.to_numeric(long_df["time"], errors="coerce")
    if tnum.notna().mean() > 0.9:
        long_df["time"] = tnum.fillna(0).astype(int)
    else:
        long_df["time"] = pd.factorize(long_df["time"].astype(str))[0].astype(int)
    long_df["id"] = long_df["id"].astype(str)
    return long_df[["id", "time", "value"]]


def _parse_tsf(text: str) -> pd.DataFrame:
    """
    Parse Monash / forecasting.org TSF text into a wide table (first col = series id).

    Each @data line: ``series_name:start_timestamp:v1,v2,...`` (split on first two ':').
    """
    lines = text.splitlines()
    data_i = None
    for i, ln in enumerate(lines):
        if ln.strip().lower() == "@data":
            data_i = i + 1
            break
    if data_i is None:
        raise ValueError("TSF: missing @data section")
    rows: List[dict] = []
    for ln in lines[data_i:]:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split(":", 2)
        if len(parts) < 3:
            continue
        sid = parts[0].strip()
        rest = parts[2]
        vals: List[float] = []
        for x in rest.split(","):
            x = x.strip()
            if not x:
                continue
            if x == "?":
                vals.append(float("nan"))
                continue
            try:
                vals.append(float(x))
            except ValueError:
                continue
        if not vals:
            continue
        row: dict = {"id": sid}
        for j, v in enumerate(vals):
            row[f"t{j}"] = v
        rows.append(row)
    if not rows:
        raise ValueError("TSF: no series parsed from @data")
    wide = pd.DataFrame(rows)
    cols = ["id"] + [c for c in wide.columns if c != "id"]
    return wide[cols]


def _read_forecasting_zip_table(zip_path: Path) -> pd.DataFrame:
    """Read the main table inside a Monash release zip (.tsf or .csv)."""
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = [n for n in z.namelist() if "__MACOSX" not in n]
            stem = zip_path.stem.lower().replace(" ", "_")
            tsf = [n for n in names if n.lower().endswith(".tsf")]
            csv = [n for n in names if n.lower().endswith(".csv")]
            if tsf:
                pick = [n for n in tsf if Path(n).stem.lower() == stem] or sorted(tsf, key=len)
                target = pick[0]
                text = z.read(target).decode("utf-8", errors="replace")
                return _parse_tsf(text)
            if not csv:
                raise FileNotFoundError(f"No .tsf or .csv in {zip_path}")
            return _read_csv_from_zip(zip_path, prefer=None)
    except zipfile.BadZipFile as e:
        raise RuntimeError(
            f"Corrupt or incomplete ZIP (common after a partial download): {zip_path}. "
            "Delete the file, re-download from the URL in data/download_all.sh, "
            "and verify with: python -m zipfile -t <path>"
        ) from e


def _read_csv_from_zip(zip_path: Path, prefer: Optional[str] = None) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.endswith(".csv") and "__MACOSX" not in n]
        if not names:
            raise FileNotFoundError(f"No .csv inside {zip_path}")
        # Prefer CSV whose basename matches the zip stem (Monash releases).
        stem = zip_path.stem.lower().replace(" ", "_")
        stem_hits = [n for n in names if Path(n).stem.lower() == stem]
        if stem_hits:
            names = stem_hits
        elif prefer:
            names = [n for n in names if prefer in n.replace("\\", "/").lower()] or names
        # drop obvious readme / metadata
        names = [n for n in names if "readme" not in n.lower()]
        names.sort(key=lambda x: len(x))
        target = names[0]
        with z.open(target) as fh:
            # skip leading junk rows starting with @ (some Monash releases)
            text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    start = 0
    while start < len(lines) and (not lines[start] or lines[start].startswith("@")):
        start += 1
    body = "\n".join(lines[start:])
    return pd.read_csv(io.StringIO(body))


def _read_first_xlsx_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        xlsxs = [
            n
            for n in z.namelist()
            if n.lower().endswith(".xlsx") and "__MACOSX" not in n
        ]
        if not xlsxs:
            raise FileNotFoundError(f"No .xlsx inside {zip_path}")
        xlsxs.sort(key=len)
        with z.open(xlsxs[0]) as fh:
            return pd.read_excel(io.BytesIO(fh.read()), engine="openpyxl")


def _abs_xlsx_to_long(path: Path) -> pd.DataFrame:
    """Heuristic: first sheet, entity in first column, many numeric columns -> melt."""
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError(f"{path}: need at least 2 columns")
    first = str(df.columns[0])
    num_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 4:
        long_df = df[[first] + num_cols].melt(id_vars=[first], var_name="time", value_name="value")
        long_df.rename(columns={first: "id"}, inplace=True)
    else:
        # stack numeric columns only, use row index as id
        num = df.select_dtypes(include=["number"])
        if num.shape[1] == 0:
            raise ValueError(f"{path}: no numeric columns found")
        long_df = num.reset_index().melt(id_vars="index", var_name="time", value_name="value")
        long_df.rename(columns={"index": "id"}, inplace=True)
        long_df["id"] = long_df["id"].astype(str)
    tnum = pd.to_numeric(long_df["time"], errors="coerce")
    if tnum.notna().mean() > 0.9:
        long_df["time"] = tnum.fillna(0).astype(int)
    else:
        long_df["time"] = pd.factorize(long_df["time"].astype(str))[0].astype(int)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])
    long_df["id"] = long_df["id"].astype(str).str.replace(r"\s+", "_", regex=True)
    return long_df[["id", "time", "value"]]


def _subsample_ids(df: pd.DataFrame, max_series: int, seed: int = 0) -> pd.DataFrame:
    if max_series <= 0 or df["id"].nunique() <= max_series:
        return df
    rng = pd.Series(df["id"].unique()).sample(min(max_series, df["id"].nunique()), random_state=seed)
    return df[df["id"].isin(rng)]


# ---------------------------------------------------------------------------
# Per-dataset builders
# ---------------------------------------------------------------------------


def _abs_data1_long(path: Path, sheet_name: str = "Data1") -> pd.DataFrame:
    """
    ABS / labour-style ``Data1`` sheet: metadata rows 0--8, row 9 = series IDs,
    column 0 from row 10 = time index, numeric columns = series values.
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    meta_end = 10
    rows: List[dict] = []
    n = df.shape[1]
    for c in range(1, n):
        sid = str(df.iloc[9, c]).strip()
        if not sid or sid.lower() == "nan":
            continue
        for i, r in enumerate(range(meta_end, len(df))):
            v = df.iloc[r, c]
            if pd.isna(v):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            rows.append({"id": f"labor/{sid}", "time": i, "value": fv})
    if not rows:
        raise ValueError(f"{path}: Data1 produced no numeric rows")
    return pd.DataFrame(rows)


def _abs_prison_table1_long(path: Path) -> pd.DataFrame:
    """Prison ABS ``Table_1`` summary (skiprows=4 layout with states as columns)."""
    df = pd.read_excel(path, sheet_name="Table_1", header=None, skiprows=4, engine="openpyxl")
    states: List[str] = []
    col_idx: List[int] = []
    for j in range(1, df.shape[1]):
        st = df.iloc[0, j]
        if pd.isna(st):
            continue
        states.append(str(st).strip())
        col_idx.append(j)
    rows: List[dict] = []
    category = "unknown"
    for r in range(2, len(df)):
        cell0 = df.iloc[r, 0]
        if pd.isna(cell0):
            continue
        s0 = str(cell0).strip()
        if "Qtr" in s0 or s0[:3] in ("Mar", "Jun", "Sep", "Dec"):
            quarter = s0
            for j, st in zip(col_idx, states):
                v = df.iloc[r, j]
                if pd.isna(v):
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                sid = f"prison/{_slug(category)}/{_slug(st)}"
                rows.append({"id": sid, "time": quarter, "value": fv})
        else:
            category = s0
    if not rows:
        raise ValueError(f"{path}: Table_1 produced no rows")
    out = pd.DataFrame(rows)
    out["time"] = pd.factorize(out["time"])[0].astype(int)
    return out[["id", "time", "value"]]


def _tourism_sheet_to_long(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """ABS tourism cube sheet: find header row of year/period columns, then melt."""
    best_r: Optional[int] = None
    best_score = 0
    for r in range(min(18, len(df))):
        score = 0
        for c in range(1, min(df.shape[1], 80)):
            v = df.iloc[r, c]
            if isinstance(v, (int, float)) and pd.notna(v):
                score += 1
                continue
            if pd.isna(v):
                continue
            s = str(v).strip()
            if len(s) >= 4 and (s[:4].isdigit() or (len(s) >= 7 and s[2] == "-" and s[5] == "-")):
                score += 1
        if score > best_score:
            best_score = score
            best_r = r
    if best_r is None or best_score < 2:
        return pd.DataFrame()
    hr = int(best_r)
    rows: List[dict] = []
    for r in range(hr + 2, len(df)):
        rid = df.iloc[r, 0]
        if pd.isna(rid):
            continue
        rid_s = str(rid).strip()
        if not rid_s or rid_s.lower() == "nan":
            continue
        for c in range(1, df.shape[1]):
            v = df.iloc[r, c]
            if pd.isna(v):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            hc = df.iloc[hr, c]
            tid = f"{prefix}/{_slug(rid_s)}"
            rows.append({"id": tid, "time": str(hc), "value": fv})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["time"] = pd.factorize(out["time"])[0].astype(int)
    return out[["id", "time", "value"]]


def _tourism_zip_to_long(zip_path: Path) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".xlsx") or "__MACOSX" in name:
                continue
            buf = io.BytesIO(zf.read(name))
            xf = pd.ExcelFile(buf, engine="openpyxl")
            for sn in xf.sheet_names:
                if "Table" not in sn:
                    continue
                df = pd.read_excel(xf, sn, header=None, engine="openpyxl")
                prefix = f"{Path(name).stem}/{_slug(sn)}"
                sub = _tourism_sheet_to_long(df, prefix=prefix)
                if len(sub):
                    parts.append(sub)
    if not parts:
        raise ValueError("Tourism-AU: no Table* sheets produced numeric rows")
    return pd.concat(parts, ignore_index=True)


def _abs_xlsx_to_long_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError("ABS sheet: need >= 2 columns")
    first = str(df.columns[0])
    num_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 2:
        long_df = df[[first] + num_cols].melt(id_vars=[first], var_name="time", value_name="value")
        long_df.rename(columns={first: "id"}, inplace=True)
    else:
        num = df.select_dtypes(include=["number"])
        long_df = num.reset_index().melt(id_vars="index", var_name="time", value_name="value")
        long_df.rename(columns={"index": "id"}, inplace=True)
        long_df["id"] = long_df["id"].astype(str)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])
    long_df["id"] = long_df["id"].astype(str).str.replace(r"\s+", "_", regex=True)
    long_df["time"] = pd.factorize(long_df["time"])[0]
    return long_df[["id", "time", "value"]]


def preprocess_tourism(raw_root: Path, max_series: int) -> pd.DataFrame:
    z = _first_match(
        raw_root,
        [
            "tourism_AU_abs_2023-24_data-cubes-all.zip",
        ],
    )
    if z is not None:
        df = _tourism_zip_to_long(z)
    else:
        x = _first_match(raw_root, ["**/*tourism*.xlsx"])
        if x is None:
            raise FileNotFoundError(
                "Tourism-AU: add tourism_AU_abs_2023-24_data-cubes-all.zip to data/raw "
                "or an extracted .xlsx"
            )
        df = _abs_xlsx_to_long(x)
    return _subsample_ids(df, max_series)


def preprocess_labor(raw_root: Path, max_series: int) -> pd.DataFrame:
    p = _first_match(
        raw_root,
        [
            "labour_AU_abs_jun-2025_table-1_total-all-industries.xlsx",
            "**/labour*.xlsx",
        ],
    )
    if p is None:
        raise FileNotFoundError("Labour-AU: add ABS June 2025 labour xlsx to data/raw")
    try:
        df = _abs_data1_long(p, sheet_name="Data1")
    except Exception:
        df = _abs_xlsx_to_long(p)
    return _subsample_ids(df, max_series)


def preprocess_prison(raw_root: Path, max_series: int) -> pd.DataFrame:
    p = _first_match(
        raw_root,
        [
            "prison_AU_abs_jun-quarter-2023_corrective-services.xlsx",
            "**/prison*.xlsx",
            "**/Corrective*.xlsx",
        ],
    )
    if p is None:
        raise FileNotFoundError("Prison-AU: add ABS corrective services xlsx to data/raw")
    try:
        df = _abs_prison_table1_long(p)
    except Exception:
        df = _abs_xlsx_to_long(p)
    return _subsample_ids(df, max_series)


def preprocess_monash_zip(raw_root: Path, zip_name: str, inner_hint: Optional[str], max_series: int) -> pd.DataFrame:
    zp = raw_root / zip_name
    if not zp.is_file():
        raise FileNotFoundError(f"Missing {zp} (run data/download_all.sh)")
    wide = _read_forecasting_zip_table(zp)
    long_df = _wide_first_col_to_long(wide)
    return _subsample_ids(long_df, max_series)


def _wiki_loose_tsf_to_long(path: Path, max_series: int, seed: int = 0) -> pd.DataFrame:
    """
    Stream a Monash TSF file from disk in two passes so ``--max-series`` caps RAM
    (full-file wide+melt can require tens of GiB for Kaggle web traffic).
    """
    ids: List[str] = []
    in_data = False
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            s = ln.strip()
            if not in_data:
                if s.lower() == "@data":
                    in_data = True
                continue
            if not s or s.startswith("#"):
                continue
            parts = s.split(":", 2)
            if len(parts) >= 3:
                ids.append(parts[0].strip())
    uniq = list(dict.fromkeys(ids))
    if not uniq:
        raise ValueError(f"{path}: TSF has no @data series rows")
    if max_series > 0 and len(uniq) > max_series:
        keep = set(pd.Series(uniq).sample(min(max_series, len(uniq)), random_state=seed))
    else:
        keep = set(uniq)

    rows: List[dict] = []
    in_data = False
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            s = ln.strip()
            if not in_data:
                if s.lower() == "@data":
                    in_data = True
                continue
            if not s or s.startswith("#"):
                continue
            parts = s.split(":", 2)
            if len(parts) < 3:
                continue
            sid = parts[0].strip()
            if sid not in keep:
                continue
            rest = parts[2]
            for t, seg in enumerate(rest.split(",")):
                x = seg.strip()
                if not x or x == "?":
                    continue
                try:
                    v = float(x)
                except ValueError:
                    continue
                rows.append({"id": sid, "time": t, "value": v})
    if not rows:
        raise ValueError(f"{path}: no numeric points after streaming parse")
    return pd.DataFrame(rows)


def preprocess_wiki(raw_root: Path, max_series: int) -> pd.DataFrame:
    """Kaggle web traffic: prefer extracted Monash TSF (``.tsf`` / ``.ts``) over the Zenodo zip."""
    loose = _first_match(
        raw_root,
        [
            "kaggle_web_traffic_dataset_with_missing_values.tsf",
            "kaggle_web_traffic_dataset_with_missing_values.ts",
        ],
    )
    if loose is not None:
        if max_series > 0:
            return _wiki_loose_tsf_to_long(loose, max_series)
        text = loose.read_text(encoding="utf-8", errors="replace")
        wide = _parse_tsf(text)
        long_df = _wide_first_col_to_long(wide)
        return long_df
    return preprocess_monash_zip(
        raw_root,
        "kaggle_web_traffic_dataset_with_missing_values.zip",
        "kaggle",
        max_series,
    )


def preprocess_m5(raw_root: Path, max_series: int, m5_max_rows: int) -> pd.DataFrame:
    candidates = [
        "m5/sales_train_evaluation.csv",
        "m5/sales_train_validation.csv",
        "sales_train_evaluation.csv",
    ]
    path = None
    for c in candidates:
        p = raw_root / c
        if p.is_file():
            path = p
            break
        pz = raw_root / (c + ".zip")
        if pz.is_file():
            path = pz
            break
    if path is None:
        raise FileNotFoundError(
            "M5: expected data/raw/m5/sales_train_evaluation.csv(.zip) (Kaggle download)"
        )

    id_cols = ["state_id", "store_id", "cat_id", "dept_id", "item_id"]
    df = pd.read_csv(path, nrows=m5_max_rows)
    missing = [c for c in id_cols if c not in df.columns]
    if missing:
        raise ValueError(f"M5: missing columns {missing}")
    d_cols = [c for c in df.columns if re.match(r"d_\d+", c)]
    if not d_cols:
        raise ValueError("M5: no d_* day columns found")
    df["id"] = df[id_cols].astype(str).agg("/".join, axis=1)
    long_df = df[["id"] + d_cols].melt(id_vars=["id"], var_name="time", value_name="value")
    long_df["time"] = long_df["time"].str.replace("d_", "", regex=False).astype(int)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").fillna(0.0)
    return _subsample_ids(long_df, max_series)


def preprocess_logicgraph(raw_root: Path, max_series: int) -> pd.DataFrame:
    base = _first_match(
        raw_root,
        [
            "logicgraph/LogicGraph/data",
            "LogicGraph/data",
        ],
    )
    if base is None or not base.is_dir():
        raise FileNotFoundError(
            "LogicGraph: clone under data/raw/logicgraph/LogicGraph (see download_all)"
        )
    rows: List[dict] = []
    for complete in sorted(base.rglob("complete.json")):
        rel = complete.relative_to(base).as_posix().replace("/complete.json", "")
        try:
            data = json.loads(complete.read_text(encoding="utf-8"))
        except Exception:
            continue
        kb = data.get("knowledge_base") or {}
        rules = kb.get("rules") or []
        facts = kb.get("facts") or []
        seq: List[dict] = []
        if isinstance(rules, list):
            seq.extend(rules)
        if isinstance(facts, list):
            seq.extend(facts)
        if not seq:
            continue
        for t, item in enumerate(seq):
            expr = item.get("prover9_expression") or item.get("formal") or ""
            if not isinstance(expr, str):
                expr = str(expr)
            rows.append({"id": rel, "time": t, "value": float(len(expr))})
    if not rows:
        raise RuntimeError("LogicGraph: no complete.json produced rows; check clone path")
    out = pd.DataFrame(rows)
    return _subsample_ids(out, max_series)


def preprocess_medical(raw_root: Path, max_series: int) -> pd.DataFrame:
    p = _first_match(
        raw_root,
        [
            "med_diag_path/healthbranchesqa/hb_dataset.csv",
            "**/hb_dataset.csv",
        ],
    )
    if p is None:
        raise FileNotFoundError("Med-Diag-Path: add healthbranchesqa/hb_dataset.csv under data/raw")
    df = pd.read_csv(p)
    path_col = "path" if "path" in df.columns else "old_path"
    rows: List[dict] = []
    for _, row in df.iterrows():
        raw_path = str(row.get(path_col, "") or "")
        parts = [x.strip() for x in raw_path.replace("\n", " ").split("->") if x.strip()]
        if not parts:
            continue
        sid = "/".join(parts)
        for t, _seg in enumerate(parts):
            rows.append({"id": sid, "time": t, "value": float(len(_seg))})
    if not rows:
        raise RuntimeError("Medical: no rows from path column")
    out = pd.DataFrame(rows)
    return _subsample_ids(out, max_series)


DATASET_ORDER = (
    "tourism",
    "labor",
    "prison",
    "m5",
    "wiki",
    "electricity",
    "traffic",
    "solar",
    "logic",
    "medical",
)


def build_dataset(name: str, raw_root: Path, max_series: int, m5_max_rows: int) -> pd.DataFrame:
    name = name.lower().strip()
    if name == "tourism":
        return preprocess_tourism(raw_root, max_series)
    if name in {"labor", "labour"}:
        return preprocess_labor(raw_root, max_series)
    if name == "prison":
        return preprocess_prison(raw_root, max_series)
    if name == "m5":
        return preprocess_m5(raw_root, max_series, m5_max_rows)
    if name == "wiki":
        return preprocess_wiki(raw_root, max_series)
    if name == "electricity":
        return preprocess_monash_zip(raw_root, "electricity_hourly_dataset.zip", "electricity", max_series)
    if name == "traffic":
        return preprocess_monash_zip(raw_root, "traffic_hourly_dataset.zip", "traffic", max_series)
    if name == "solar":
        return preprocess_monash_zip(raw_root, "solar_10_minutes_dataset.zip", "solar", max_series)
    if name in {"logic", "logicgraph"}:
        return preprocess_logicgraph(raw_root, max_series)
    if name in {"medical", "med"}:
        return preprocess_medical(raw_root, max_series)
    raise ValueError(f"unknown dataset: {name}")


def default_out_name(name: str) -> str:
    n = name.lower().strip()
    if n in {"labour"}:
        return "labor"
    if n in {"logicgraph"}:
        return "logic"
    if n in {"med"}:
        return "medical"
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess Table-6 datasets to long CSV.")
    ap.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"One of: {', '.join(DATASET_ORDER)}, all",
    )
    ap.add_argument("--raw-root", type=str, default="data/raw", help="Root folder with downloads")
    ap.add_argument("--out-dir", type=str, default="data/processed", help="Output directory for CSVs")
    ap.add_argument(
        "--max-series",
        type=int,
        default=2000,
        help="Max unique series ids per dataset (<=0 means no cap)",
    )
    ap.add_argument(
        "--m5-max-rows",
        type=int,
        default=5000,
        help="Read at most this many rows from M5 sales CSV (speed)",
    )
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not raw_root.is_dir():
        raise SystemExit(f"raw-root is not a directory: {raw_root}")

    names = list(DATASET_ORDER) if args.dataset.lower() == "all" else [args.dataset]

    for name in names:
        oname = default_out_name(name)
        out_path = out_dir / f"{oname}.csv"
        try:
            df = build_dataset(oname, raw_root, args.max_series, args.m5_max_rows)
        except Exception as e:
            print(f"[skip] {oname}: {e}")
            continue
        if df is None or len(df) == 0:
            print(f"[skip] {oname}: empty dataframe after preprocessing")
            continue
        _ensure_out(out_path)
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}  rows={len(df)}  series={df['id'].nunique()}")


if __name__ == "__main__":
    main()
