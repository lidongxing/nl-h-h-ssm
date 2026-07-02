#!/usr/bin/env bash
# Optional: split docs/descriptions/ into per-folder commits so GitHub shows
# distinct "Last commit message" text for each description stub batch.
#
#   bash scripts/commit_doc_layers.sh
#
# Prerequisite: run `python scripts/generate_repo_manifest.py` first.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

git add REPO_MANIFEST.json README.md docs/README.md scripts/generate_repo_manifest.py results/README.md
git commit -m "docs: add REPO_MANIFEST.json with unique per-path descriptions" || true

declare -A MSGS=(
  [nlh_ssm]="docs(nlh_ssm): hyperbolic ops, NL-H block, loader, sMAPE/RMSSE/ACD metrics"
  [csrc]="docs(csrc): PH-Scan kernel and Cauchy helper notes"
  [models]="docs(models): nlh_ssm, ph_scan, acg_gate facades"
  [experiments]="docs(experiments): run_exp CLI and per-dataset YAML configs"
  [configs]="docs(configs): Table 6 nlh_tuned.yaml protocol fields"
  [data]="docs(data): download, preprocess, committed CSV stems"
  [scripts]="docs(scripts): Table 6 campaign, LaTeX, RQ3 orchestration"
  [benchmarks]="docs(benchmarks): speed JSON, ablation, sensitivity, figure scripts"
  [results]="docs(results): 50 Table 6 JSON logs with distinct metrics blurbs"
  [tests]="docs(tests): pytest hyperbolic, PH-Scan, metrics"
  [assets]="docs(assets): generated figure outputs"
)

for dir in "${!MSGS[@]}"; do
  if [[ -d "docs/descriptions/${dir}" ]]; then
    git add "docs/descriptions/${dir}"
    git commit -m "${MSGS[$dir]}" || true
  fi
done

# Root-level description stubs (.gitignore, setup.py, README.md, etc.)
git add docs/descriptions/*.md 2>/dev/null || true
git commit -m "docs(root): description stubs for setup.py, README, git config" || true

echo "Done. Run: git push origin main"
