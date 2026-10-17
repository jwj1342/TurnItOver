# Kimi Level 0 fixed-evidence axis diagnosis

This is a one-shot `k3-256k` diagnostic on the ten committed four-panel inputs from
`f34b07c127266b401564592bfe38bb6c9e7121fb`: five independent Train assets, each
with one clean and one `joint_axis` case. Before inference, all ten completed human
reviews were matched against private Gold and the independent checker, and every input
PNG SHA-256 was recorded. The public pending manifest was not used to select cases.

Each request sent exactly one four-panel PNG and the same short prompt. It contained no
Gold, human answer, asset ID, corruption parameter, or few-shot example. There was one
call per case, no retries, no prompt changes, no training, and no Test access. Per-case
directories retain the copied input, PNG SHA, prompt, redacted-metadata request record,
exact API request body, raw response, result, and reported usage. Gold was joined only
after all calls ended.

## Results

| group | correct pass/fail | errors | image-consistent motion explanation |
| --- | ---: | ---: | ---: |
| clean | 5/5 | 0 | 5/5 |
| joint_axis | 5/5 | 0 | 5/5 |
| total | 10/10 | 0 | 10/10 |

All outputs followed the two-line contract. Token usage was 5,570 prompt, 15,456
completion (14,988 reasoning), and 21,026 total tokens; gateway billing remains
unknown. The manual visual audit in `motion_explanation_review.json` explicitly checks
the descriptions against the panels. In particular, Kimi correctly described the USB
clean as sliding and the USB axis case as a horizontal reference trajectory versus a
downward folded candidate, rather than treating every case as rotation.

The fixed paired-tile pixel baseline (`fail` iff max rest/limit MAE exceeds 0.005) also
gets 10/10. It therefore establishes only that every axis case has a visible mismatch;
it cannot state a motion direction. Kimi's 10/10 one-sentence comparison is evidence
of direction-level success on this deliberately extreme Level 0 slice, but not evidence
of robustness beyond these five assets or of causal joint-mechanics understanding.

## Decision

This bundle is a suitable **starting gate** for later teacher diagnosis: labels and
inputs are independently checked, defects are visually obvious, and the model clears
the fixed protocol. It is not sufficient by itself for SFT labels or a general teacher
claim, because the simple pixel baseline is equally accurate and the sample has only
five assets. Keep it frozen as a diagnostic gate, then add subtler and pixel-matched
controls before using any teacher output for supervision.
