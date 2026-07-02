#!/usr/bin/env bash
# Downloads public Monash Forecasting Repository zips (subset of Table 3).
# Requires curl (recommended) or wget. On Windows without bash, use:
#   powershell: see README or run each curl.exe line manually.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data/raw"
mkdir -p "${DEST}"

download() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "${out}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${out}" "${url}"
  else
    echo "Error: need curl or wget." >&2
    exit 1
  fi
}

echo "Downloading to ${DEST} ..."

# --- Socio-economic (ABS official sources, aligned with paper citations) ---
# Tourism-AU (ABS2024Tourism): 2023-24 data cubes bundle
download "https://www.abs.gov.au/statistics/economy/national-accounts/tourism-satellite-account/2023-24/Tourism-Satellite-Account-data-cubes-all.zip" \
  "${DEST}/tourism_AU_abs_2023-24_data-cubes-all.zip"

# Labour-AU (ABS2025Labour): June 2025, table 1 (total all industries)
download "https://www.abs.gov.au/statistics/labour/labour-accounts/labour-account-australia/jun-2025/6150055003DO001.xlsx" \
  "${DEST}/labour_AU_abs_jun-2025_table-1_total-all-industries.xlsx"

# Prison-AU (ABCS2023Prison): use ABS Corrective Services June quarter 2023 datacube
download "https://www.abs.gov.au/statistics/people/crime-and-justice/corrective-services-australia/jun-quarter-2023/Corrective%20Services%2C%20Australia%20-%20June%20quarter%202023.xlsx" \
  "${DEST}/prison_AU_abs_jun-quarter-2023_corrective-services.xlsx"

# --- Industrial & IoT (HTS-oriented benchmark sources) ---
# NOTE:
# - We keep project dataset IDs/names stable (m5/wiki/electricity/traffic/solar)
#   and only standardize download provenance here.
# - For paper citations, prefer dataset-release sources over model papers.
#
# M5-Walmart: canonical source is Kaggle competition (Makridakis et al. 2022 context).
# Requires: pip install kaggle && ~/.kaggle/kaggle.json
# kaggle competitions download -c m5-forecasting-accuracy -p "${DEST}/m5"
#
# Wiki-Traffic: use Monash "Kaggle Web Traffic" bundle on Zenodo for reproducible HTS ingest.
download "https://zenodo.org/records/4656080/files/kaggle_web_traffic_dataset_with_missing_values.zip" \
  "${DEST}/kaggle_web_traffic_dataset_with_missing_values.zip"

# Electricity-L / Traffic-HTS / Solar-HTS:
# use Monash hierarchical benchmark releases (Zenodo mirrors from forecastingdata.org links).
download "https://zenodo.org/records/4656140/files/electricity_hourly_dataset.zip" \
  "${DEST}/electricity_hourly_dataset.zip"
download "https://zenodo.org/records/4656132/files/traffic_hourly_dataset.zip" \
  "${DEST}/traffic_hourly_dataset.zip"
download "https://zenodo.org/records/4656144/files/solar_10_minutes_dataset.zip" \
  "${DEST}/solar_10_minutes_dataset.zip"

# --- Expert systems (synthetic / private) ---
# LogicGraph / Expert-Log: generate locally.
# Med-Diag-Path: add your approved clinical data source.

echo "Done. Unzip archives under data/raw/ or data/processed/ as needed."
