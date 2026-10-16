# PartNet-Mobility obvious-defect review bundle

This bundle contains nine blind comparison boards: three Train assets, each with three candidate variants. Every board remains `pending_human_review`; automated gates do not promote a candidate to training data. Gold labels, generation parameters, independent checker details, and rejected attempts are under `private/`. No Kimi/model call or SFT training is part of this workflow, and Test is not read or modified.

Each board shows reference and candidate at rest, mid, and limit using native 256 x 256 Chromium tiles. It names the asset, selected joint, joint type, moving part, and view, but does not show the defect answer. Fill `human_review.jsonl` only after deciding whether the error type, joint, part, and evidence frames are unambiguous.

## Reproduce

Run from the repository root into a fresh output directory:

```bash
repro_root=$(mktemp -d /tmp/turnitover-obvious-defects-repro.XXXXXX)
python -m scripts.prepare_obvious_defect_review --out "$repro_root/bundle"
diff -rq docs/evidence/partnet-mobility-obvious-defects-v1 "$repro_root/bundle"
```

## Validate

```bash
python -m pytest tests/unit/test_obvious_defect_review.py -q
python -m pytest tests/unit -q
python -m py_compile scripts/prepare_obvious_defect_review.py
git diff --check
```
