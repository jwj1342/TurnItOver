# Kimi Easy Object active-teacher collection

This run uses exactly the 18 conservatively selected cases in the completed human
audit: seven clean, six axis defects, five part offsets, seven independent Train
assets. No prior 117-row dataset is merged. No training is launched.

The declared task is the normal-selected joint at the original human-reviewed
camera. The teacher chooses candidate observations, joint detents, and when to
finish; the full camera set and other joints are outside this bounded scope.
Reference rest/mid/limit images and one initial public hierarchy query are common
inputs. No candidate image, fixed motion action, Gold, human answer, corruption
parameter, checker result, or private screening metric is supplied before the
teacher acts. At most seven teacher-selected observations and one terminal call
fit the eight-call Episode cap. All calls are serial, with no schema retries,
automatic network retries, episode retries, fallback, or per-failure prompt tuning.
Unknown gateway billing remains null.

From repository root, into a fresh output directory:

```bash
python -m scripts.run_kimi_easy_interactive --phase prepare --out /tmp/kimi-easy-repro
python -m scripts.run_kimi_easy_interactive --phase pilot --out /tmp/kimi-easy-repro
# Inspect the first Episode and its audit before the remaining paid collection.
python -m scripts.run_kimi_easy_interactive --phase remaining --out /tmp/kimi-easy-repro
python -m scripts.run_kimi_easy_interactive --phase finalize --out /tmp/kimi-easy-repro
```

The pilot must pass pipeline validation before `remaining` runs. A failed
diagnosis is retained and does not trigger a retry. Existing paid Episode
directories are rejected. `prepare` renders and checks all candidates before any
API call, including exact pixel comparison with the human-reviewed board tiles.
`finalize` is offline and exports to a fresh `sft/` directory.

Offline loader/processor validation (no model weights or training):

```bash
../TurnItOver/.venv-qwen/bin/python -m scripts.check_teacher_export_loading \
  --dataset docs/evidence/kimi-easy-interactive-v2/sft \
  --out docs/evidence/kimi-easy-interactive-v2/processor_validation.json \
  --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --all-processor-rows
```

The original single-reference pass auditor is extended locally only for the
predeclared motion scope. A clean pass must cover every distinct normal reference
state with actual frozen-bound candidate pixels, and the independent checker must
confirm the object is clean. Defect findings must match private truth and cite
perceptible, geometry-confirmed, reference-comparable observations. These rules
are frozen before the pilot. An uncertain diagnosis is retained separately and
never rewritten into a correct diagnosis target. Valid acquisition actions are
qualified independently of the terminal diagnosis, using new, bound, comparable
states; this does not certify that an action is optimal.

The exporter copies the exact teacher-visible prompt and ordered image list for
each causal prefix. The real SFT loader and the additional parity validator check
image SHA, prompt identity, target identity, and source provenance. Private
auditing fields remain outside model messages.

Tests:

```bash
python -m pytest tests/unit/test_easy_teacher_audit.py tests/unit/test_teacher.py \
  tests/unit/test_teacher_evidence.py tests/unit/test_teacher_export.py -q
python -m pytest tests/browser/test_teacher.py -m browser -q
git diff --check
```

Do not run the collection commands against this saved output to replace an
unfavorable result. The raw prompts/responses, causal trajectories, images and
metadata are immutable experiment records; corrected posthoc audit versions, if
needed, must preserve their original files.
