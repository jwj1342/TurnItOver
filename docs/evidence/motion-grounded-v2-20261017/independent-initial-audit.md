# Motion-grounded v2 independent integrity audit

Date: 2026-10-17. Reviewer: fresh Codex agent. `review_independence: same-family`; `acceptance_status: provisional`.

Overall **FAIL for the old frozen full/evidence metrics**, with verified real training artifacts. This initial report remains open for the new pixel-only run results; it does not claim completed new training/evaluation.

## A. Ground-truth provenance — WARN

No inspected target is generated from model predictions. The source renderer replays dataset corruptions and separates axis, offset, and clean (`scripts/build_motion_diagnosis.py:112-146`). Main supervision combines those construction labels with a deterministic visibility gate: static/temporal RGB features and a 0.001 threshold determine comparison and observability; training objects are then selected to make signatures agree with labels (`scripts/build_motion_grounded.py:56-76`, `:94-109`; `turnitover/oracle/motion_grounding.py:46-65`). Therefore comparison accuracy tests a deterministic image proxy. High classification on these selected rows does not establish 3D axis-versus-offset semantics.

The main validation data retain two signature/label disagreements: `PM_100202:offset:full:full_motion` is proxy-clean but target-uncertain; `PM_102252:offset:full:full_motion` is proxy-axis but target-offset (`output/motion-grounded-v2-intermediate/validation.jsonl:3`, `:9`). This is not model-derived fake GT, but latent defect, observation sufficiency, and pixel signature must be reported separately.

Frozen B/C classes have stronger independent semantic provenance: retained single corruptions, parameter-difference assertions, browser world-vertex checks (`scripts/build_motion_attribution.py:76-96`), and separately recorded SVD rigid fitting (`output/motion-attribution-v1/geometry.audit.json:234`). This is simulated semantic GT, not human annotation. Geometry validates the defect but cannot alone establish which image supports a citation.

## B. Normalization — PASS

The inspected evaluator computes count divided by row count, without normalization by predictions (`scripts/evaluate_qwen_motion_diagnosis.py:70-80`). All six stored rates across all five old evaluation directories independently recompute exactly from predictions. Comparison fields are raw counts. Null-ID accuracy must still be interpreted carefully: invalid predictions become `{}`, so missing IDs can compare equal to null gold (`:37-41`).

## C. Artifact existence, identity, and real training — WARN pending new results

Old intermediate and direct each have 108 examples, 2 epochs, 216 sequential finite-loss steps, 108 distinct training IDs and a 12,878,000-byte adapter. Both adapter SHA values and dataset manifest/train SHA values match their run records (`output/motion-grounded-v2-model/run.json:1`, `output/motion-grounded-v2-control/run.json:1`). Model hashes: intermediate `629ff4ca254d0edd3098f565919c42eef27ca3ebd337d257fe0059b5a97593b1`; direct `455af83ca605fbb0589c945da301be42c60aec1c8873a8a688b795a08e7d39a7`.

Old main datasets each have 443 unique PNGs / 1,656 references with no SHA mismatch; final suites each have 182 referenced unique PNGs / 1,296 references with no SHA mismatch. Old suite metric manifest hashes match the `*-suite-final` versions, not the earlier suite manifests.

New `direct-panels` and `intermediate-panels` independently pass pairing: 108 Train / 30 Validation rows, byte-identical public prompt, image hashes, IDs, and diagnosis targets. Only the intermediate assistant target adds the original comparison. No `rgb_comparison` is model-visible. All 552 image references / 458 unique PNGs per arm match SHA. All 1,656 source tiles exactly equal their claimed location/view/step in the new panels. The corrected builder hashes the entire tile sequence (`scripts/prepare_motion_pixel_control.py:40-47`). The earlier `direct-data` / `intermediate-data` each have 16 mismatching image references caused by overwritten panel names; keep them as failed artifacts and exclude them from valid training/results.

## D. Metrics and schema — FAIL in old artifacts

1. **Four frozen offset rows cannot pass full evidence under the current proxy rule.** Final suite lines 125, 126, 143, and 144 are PM_100202/PM_1280 offsets in B/C. Their class is forced to the original semantic `offset`, but both `evidence_steps` and proxy `static_steps` are empty. `scripts/prepare_motion_grounded_eval.py:47-50` changes the class without restoring original evidence, while `scripts/evaluate_qwen_motion_diagnosis.py:51-52` requires a nonempty subset of proxy static steps. The same defect is inherited by initial new transfer data.
2. The old axis evidence override only checks whether one pair is a subset, permitting extra nonexistent/noninteger/duplicate steps (`scripts/evaluate_qwen_motion_diagnosis.py:46-50` in the pre-repair version). The live executor has added a legal-step check; new results must bind the repaired scorer, and old results must be rescored separately if necessary.
3. `full_motion_n=0` in both 146-row suites is incorrect: condition names have prefixes, while code tests exact equality with `full_motion` (`scripts/evaluate_qwen_motion_diagnosis.py:72` in the current initial snapshot). Expected full-motion rows: 73.
4. All old target objects pass the four-field schema after removing optional comparison. The validator does not enforce evidence element type, legal hierarchy IDs, or comparison structure (`turnitover/oracle/motion_diagnosis.py:37-48`). Consequently `valid=1` proves only the implemented structural check. All old comparison metrics are called, but direct-arm zero comparison counts are not a fair error rate since its prompt/target never requires comparison.
5. `unsupported` is any predicted axis/offset that is not fully correct, including wrong localization and wrong type (`scripts/evaluate_qwen_motion_diagnosis.py:62`); it is not a pure unsupported-evidence count. Report those components separately.
6. Main aggregate part accuracy includes many null targets. Defect-only part results are intermediate 7/12 versus direct 5/12. Axis joint is 5/5 for both, but the exact target joint is already provided in every axis input (23/23 including train+val); this measures consistency with a supplied joint, not open joint localization.

## E. Scope and reliable cohorts — WARN

| Cohort | Rows | Objects | Interpretation |
|---|---:|---:|---|
| Main | 30 | 5 | Object-held-out; construction GT plus pixel observability proxy |
| Reverse / pixels-only / features-only | 30 each | same 5 | Ablations of the same cases, not independent evidence |
| v1_frozen | 24 | 4, all in main | Earlier rendering protocol on reused held-out objects |
| Frozen B/C | 12 each | 4, all in main | Stronger simulated semantic labels; repair citation scoring |
| Extra | 8 | 2 new | 6 uncertain + 2 clean targets, no axis or offset diagnosis positives |

Total unique held-out objects across the suite: 7; training objects: 18; disjointness verified. One seed (`20261016`). No result supports robust/open-world defect diagnosis. The extra two objects provide no new positive axis or offset recall test. Frozen means an earlier fixed Validation protocol here; the frozen Part 2 Test split is not used.

Old main class/full counts: intermediate 26/30 and 22/30; direct 27/30 and 20/30. Removing features yields 17/30 and 16/30 class; retaining features without images yields 27/30 and 26/30. These are direct evidence of feature dependence in the old models. They do not establish that intermediate text improves semantics. The new pixel-only matched-arm test is necessary to separate this dependence from learned pixel comparison.

## F. Evaluation type — simulation_only with proxy measurements

Rendered real assets plus injected single corruptions are `simulation_only`. Main category targets combine controlled construction GT and visibility proxy; intermediate comparison is `self_supervised_proxy` from RGB. Frozen B/C category/localization uses independently checked simulation semantic GT. There is no model-generated reference, natural-error Gold, or human evaluation in this audited slice.

## Necessary repairs and final claims

- Preserve original frozen files. In a new derived evaluation dataset add separately named frozen evidence metadata, with original source SHA and observed-step intersection. Use it for frozen offset support; preserve comparison fields as proxy measurements.
- Original offset support is from `gold.private.jsonl` `pixel_signals`, **not** `perceptible_steps`. Lines 2,5,8 have exposed steps `[0,1,2,3,4,5]`; line 11 has `[3,4,5]`. In C, intersect with public `[0,3]`, yielding `[0,3]` for the first three objects and `[3]` for PM_1280. Original scorer semantics are `scripts/score_motion_attribution.py:44-56`; do not call these annotations manual/human Gold.
- Fix legal citations and prefixed condition counts; bind new predictions to dataset rows, scorer hash and adapter hash. Raw predictions allow CPU-only rescoring, without rerunning inference.
- Present supplied-joint consistency, defect-only part accuracy, evidence support, class correctness and full correctness separately. Keep class and observation-sufficiency errors distinct.
- Exclude failed panel artifacts. Finish paired training/eval and update this report with actual checkpoint/loss/prediction hashes before final claims.

All currently audited input hashes are in `audit-review.json`. The audit is same-family provisional, and its deterministic artifact checks are reproducible.
