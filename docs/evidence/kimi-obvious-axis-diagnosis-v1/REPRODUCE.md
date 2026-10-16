# Kimi obvious-axis fixed-diagnosis experiment

This is a six-call, no-retry Train-only experiment: three human-approved `joint_axis`
cases and their approved clean controls. The exact prompts, visible requests, 256 px
images, raw response text, Episode ledgers, image hashes, private score, and aggregate
summary are retained here. The model received no Gold, corruption parameters, or human
answers. Billing is recorded as unknown (`null`) because the gateway price is unknown.

Preflight and run from the repository root into a fresh directory:

```bash
python -m pytest tests/unit/test_kimi_obvious_axis_diagnosis.py -q
python -m scripts.run_kimi_obvious_axis_diagnosis   --out docs/evidence/kimi-obvious-axis-diagnosis-v1   --third-party-transmission-reviewed   --transmission-review-basis user-requested-six-case-PartNet-Train-diagnostic   --estimated-input-tokens 12000
```

The command refuses an existing output directory. It makes exactly one real `k3-256k`
call per case, has no schema retry, and has a global six-request cap. Do not rerun it to
replace failed or unfavorable outputs.

Validation:

```bash
python -m pytest tests/unit/test_kimi_obvious_axis_diagnosis.py tests/unit/test_teacher.py -q
python -m pytest tests/unit -q
python -m py_compile scripts/run_kimi_obvious_axis_diagnosis.py
git diff --check
```
