#!/bin/bash
# Regenerate the LaTeX fragments of paper/vulcan_results.tex (all Vulcan results in the paper's table formats).
# Run inside a short CPU job from the repo root:  srun ... bash scripts/exp900/staging.sh
set -euo pipefail
source scripts/setup_env.sh > /dev/null 2>&1; source .venv/bin/activate
T=tio-runs; Q=$T/repair/q38; S=tio-runs/repair/q38/final/tex; G=Qwen3.8-27B-FP8; mkdir -p $S $Q/final
PACK=$T/data900/repair58/three-tier-900-validation-repair-58-v1
LABELS=$T/data900/three-tier-900-sft-v1/episode_export/annotations.private.jsonl
rows() { awk -v want="$2" '$0 ~ want {f=1; next} f && /^```$/ {if (n++) exit; next} f && n == 1' "$1" | sed -E 's/^([0-9]+B) ([a-z-]+) &/\1 \& \2 \&/'; }
# recognition, v1.1, L40S
for split in validation test; do
  python scripts/exp900/analysis/paper_recognition.py $split l40s v11 > /dev/null
  rows $T/out/RESULTS-recognition-paper-$split-l40s-v11.md "recognition_sizes.tex" > $S/rec-$split.tex
done
python scripts/exp900/analysis/table2_v11.py > /dev/null && cp $T/out/RESULTS-table2-v11.tex $S/table2.tex
cp $T/out/RESULTS-table2-v11-main.tex $S/acquisition-main.tex
python - > $S/rec-main.tex <<'PY'
# main-text recognition rows: Size & Supervision & validation All F1, Loc. F1, Support & test All F1, Loc. F1, Support
import re
def table(split):
    rows = {}
    for line in open(f"tio-runs/out/RESULTS-recognition-paper-{split}-l40s-v11.md"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 12 and re.fullmatch(r"\d+B", c[0]):
            rows[c[0], c[1]] = (c[7], c[8], c[9])
    return rows
def tex(v):
    m = re.fullmatch(r"([-\d.]+) \[([-\d.]+),([-\d.]+)\]", v)
    return f"\\rng{{{m[1]}}}{{{m[2]}}}{{{m[3]}}}" if m else v
val, test = table("validation"), table("test")
for key in sorted(val, key=lambda k: (int(k[0][:-1]), ["zero-shot", "ordinary", "evidence"].index(k[1]))):
    print(f"{key[0]} & {key[1]} & " + " & ".join(tex(v) for v in val[key] + test.get(key, ("--",) * 3)) + r" \\")
PY
cp docs/results/v11/table3.tex $S/table3.tex
# repair: per-run summaries on a case list; ours_* rows from the runs of the matching data version
summ() {  # summ <case list> <name> <out dirs...>
  local cases=$1 name=$2; shift 2
  python scripts/summarize_repair_table.py --out "$@" --cases $PACK --labels $LABELS --case-ids-file $cases --summary $Q/final/$name.json > $Q/final/$name.md
}
FIX="none pixel zs8b_screenshot zs8b_sweep zs4b_screenshot zs4b_sweep zsapi_sweep gold"
fixed() { for j in $FIX; do [ -d $Q/$1/$j ] && echo $Q/$1/$j; done; }
for v in v11 v1; do
  if [ $v = v11 ]; then C=$T/repair/eligible-v11.txt; OURS="v11-seed20261013 v11-seed20261113 v11-seed20261213"
  else C=$T/repair/eligible.txt; OURS="pilot seed20261113 seed20261213"; fi
  sums=()
  for r in pilot rep2 rep3; do summ $C $v-fixed-$r $(fixed $r); sums+=($Q/final/$v-fixed-$r.json); done
  for r in $OURS; do summ $C $v-ours-$r $(ls -d $Q/$r/ours_*); sums+=($Q/final/$v-ours-$r.json); done
  ours_runs=""; for r in $OURS; do ours_runs="$ours_runs $Q/$r:ours_2b_ordinary,ours_2b_evidence,ours_8b_evidence"; done
  fixed_runs=""; for r in pilot rep2 rep3; do fixed_runs="$fixed_runs $Q/$r:$(fixed $r | xargs -n1 basename | paste -sd,)"; done
  python scripts/exp900/analysis/repair_bootstrap.py --gen $G --cases $C --runs $fixed_runs $ours_runs --latex $S/repair-$v-success.tex --json $Q/final/bootstrap-$v.json > $Q/final/bootstrap-$v.md
  TIO_BOOTSTRAP_JSON=$Q/final/bootstrap-$v.json python scripts/exp900/analysis/paper_repair.py $G "${sums[@]}" > $S/repair-$v-rows.tex
done
python scripts/exp900/analysis/repair_examples.py $Q/pilot $G 4 > $Q/final/examples-rep1.md
date -Is > $S/GENERATED_AT.txt
for v in v11 v1; do   # split paper_repair.py output into its three tables
  awk -v p=$S/repair-$v '/^% repair.tex/{f="main"} /^% repair_breakdown.tex/{f="breakdown"} /^% repair_cost.tex/{f="cost"} /^% table1/{f="table1"} /^% per run/{f=""} f && !/^%/ {print > (p "-" f ".tex")}' $S/repair-$v-rows.tex
done
python - > $S/pixel.tex <<'PY'
import json
parts = []
for split in ("validation", "test"):
    m = json.load(open(f"tio-runs/out/level900v11-pixel/{split}/metrics.json"))
    v = m["validation"]
    parts.append(f"{split} {100 * v['status_accuracy']:.1f}\\% ({v['n']} cases, threshold {m['threshold']:g} fit on train)")
print("\\textbf{Pixel-difference status baseline (no model):} pass/fail status accuracy " + "; ".join(parts)
      + ". Clean candidates are byte-identical to their references, so status metrics do not measure understanding.")
PY
# paper style: capitalized supervision labels; no negative zero
sed -i -E 's/^([0-9]+B) & zero-shot &/\1 \& Zero-shot \&/; s/^([0-9]+B) & ordinary &/\1 \& Ordinary \&/; s/^([0-9]+B) & evidence &/\1 \& Evidence \&/' $S/rec-*.tex
sed -i -E 's/-0\.0\\,\{/0.0\\,{/g; s/\[-0\.0,/[0.0,/g' $S/repair-*-table1.tex
