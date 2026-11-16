#!/bin/bash
# Regenerate the result rows and write them into the Overleaf paper's tables (between "% rows:" markers), then push.
# Run from the repo root inside a job step:  srun --jobid=<running job> --overlap ... bash scripts/exp900/refresh_paper.sh
set -euo pipefail
bash scripts/exp900/staging.sh
P=paper
git -C $P pull -q --ff-only origin main
# tables/repair.tex is edited by hand since 11-16 (no intervals, Kimi K3 columns); its rows are in repair-v11-table1.tex
python scripts/exp900/inline_tables.py tio-runs/repair/q38/final/tex $P/tables/acquisition.tex \
  $P/tables/recognition_main.tex $P/tables/recognition_sizes.tex $P/tables/recognition_test.tex $P/tables/acquisition_full.tex \
  $P/tables/repair_breakdown.tex $P/tables/repair_cost.tex $P/tables/repair_v1.tex
git -C $P add tables
git -C $P diff --cached --quiet || { git -C $P commit -q -m "Refresh result rows from the Vulcan runs"; git -C $P push -q origin HEAD:main; }
git -C $P log --oneline -1
