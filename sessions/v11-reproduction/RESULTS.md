# Reported v1.1 results and interpretation

Source discussion: [Issue #10](https://github.com/jwj1342/TurnItOver/issues/10), [diagnosis/code handoff](https://github.com/jwj1342/TurnItOver/issues/10#issuecomment-6909908717), [final repair measurements](https://github.com/jwj1342/TurnItOver/issues/10#issuecomment-6910659110). These are the 2026-11-16 SOAR results, not results produced by publishing this package.

## Diagnosis and observations

2B and 4B each completed 6 SFT runs and 21 evaluations (Base plus three Ordinary and three Evidence seeds, across three observation conditions). Train has 600 cases; Validation admits 104; Test was not used. All models are scored against the same Evidence targets. Full-observation percentages:

| Model | Defect Macro-F1, mean [seed min, max] | Part Micro-F1 mean | Evidence support mean |
| --- | ---: | ---: | ---: |
| 2B Ordinary | 23.67 [16.37, 36.08] | 40.28 | 0.00 |
| 2B Evidence | 22.19 [12.60, 35.81] | 34.54 | 30.51 |
| 4B Ordinary | 19.07 [10.41, 29.60] | 36.30 | 0.00 |
| 4B Evidence | 19.76 [13.63, 23.47] | 36.98 | 30.82 |

Seed ranges are not confidence intervals. Defect F1 averages all five labels, including zero-positive labels as zero. Base defect F1 is zero at both sizes; evidence support without predicted findings is N/A. Validation has no uncertain-positive target; it cannot establish uncertain recall. No new 8B result is claimed.

Table 2's Viewpoint condition is Hard-only (30 cases for F1, 10 defective for visibility); Motion and Full use 104 cases for F1 and 44 defective for visibility. Visibility is 4/10, 43/44 and 44/44 respectively. Compare conditions on a common supported subset and do not inherit full-input pass certification after removing observations.

Table 3 is a separate pixel-exposure experiment over 34 admitted single-defect validation cases. Overall exposure percentages:

| K | Fixed | Random | Greedy oracle |
| ---: | ---: | ---: | ---: |
| 1 | 11.8 | 11.8 | 11.8 |
| 2 | 70.6 | 26.5 | 100.0 |
| 4 | 70.6 | 49.1 | 100.0 |
| 6 | 91.2 | 69.4 | 100.0 |

All start with the same front rest observation. Every acquisition, movement and restoration is charged. Random averages 100 orders per case before source-object bootstrap. These values do not measure learned active diagnosis, and the greedy oracle is not a globally optimal policy. The ten composite cases are excluded because the inherited exposure records do not support per-defect attribution; Structure contains only five source objects.

## Five-condition repair batch

Qwen3.8-Flash API, thinking disabled, output maximum 8,192 tokens; all 44 admitted defective validation cases, three trajectories each. Ordinary uses the v1.1 2B seed 20261013 checkpoint. Each attempt allows up to three submissions and ten total OpenHands agent steps, so an attempt can reach its step cap without making all three submissions.

| Condition | Final independent success / attempts | Success % | 95% source-object bootstrap CI % |
| --- | ---: | ---: | ---: |
| None | 2/132 | 1.52 | [0.00, 3.79] |
| Pixel difference | 5/132 | 3.79 | [0.76, 6.82] |
| Ordinary 2B | 37/132 | 28.03 | [16.67, 40.15] |
| Zero-shot qwen3.8-flash, fixed scan | 6/132 | 4.55 | [0.76, 9.09] |
| Gold, visible evidence only | 71/132 | 53.79 | [41.67, 65.15] |

Bootstrap samples 44 source objects, preserving three repetitions each; 1,000 draws, seed 20261116. These repetitions are not three training seeds. All 660 formal trajectories and documented technical reruns completed. Full Evidence repair is not part of this batch.

The independent audit checks required part identities and geometry at rest and prescribed zero/mid/limit joint states, per-part Chamfer <= 0.005 m and bbox IoU >= 0.9, plus runtime complexity bounded by the reference. Verifier pass can stop the attempt but cannot establish success. One Gold success retains an approximately 20.5 mm original offset parameter while passing those geometry thresholds; its per-injection score correctly remains unresolved. The state driver uses candidate joint limits; all 121 successful programs were checked and showed no numerical limit changes (only two -0.0 to 0.0 rewrites). Neither observation establishes correctness for every continuous state or arbitrary future rewrite.

The original input preflight passed 44/44 cases and the 660 recorded input audits had no anomalies. All generators received matching full reference/current-candidate scan definitions (4/6/12 images by tier), with fresh candidate rendering and reference-fixed framing. Gold only emits findings supported by current per-finding counterfactual pixels; it does not disclose reference source or correct numeric parameters. This remains a pixel proxy rather than human finding-level grounding.

Ordinary produced 364 saved diagnoses conforming to the output/reference contract. Zero-shot Qwen had 167 invalid diagnoses among 295; the frozen policy maps these to uncertain without format retries. Its repair result therefore includes contract-following failures, not only semantic diagnosis errors.

## Technical version history

The entire initial source-truncation batch was excluded. Formal records are in `runs-v2`, using bounded, queryable source views. Two None checker crashes due to malformed/nonfinite candidate geometry were archived and rerun after explicit failed-audit guards. Gold parser extensions and the anchor-axis-line equivalence correction were applied uniformly to the affected cases/repetitions; prior artifacts were retained, and 312 previously valid feedback records were replayed unchanged in the recorded v4 check. No rerun was chosen based on success. The final source bytes and version identities are retained in this delivery; private retry lists remain with the original experiment archive.

Per-injection restoration is scored separately using physical parameter predicates. The original 54 injections include 9 rest-visible, 42 motion-visible and 3 unexposed. One unexposed anchor displacement is already physically equivalent at initialization, gets no restoration credit, and stays in the denominator. Unsupported final-source semantics stay null; report coverage and bounds, not invented successes. Cost-to-first-success statistics are conditional on final successful trajectories. Total recorded generator-plus-verifier tokens must not be silently relabeled as generated output tokens.

## Comparison and pending work

Vulcan reported a different generator (Qwen3.8-27B-FP8, thinking on, 32,768 tokens), its own repair loop, and three training seeds rather than one fixed checkpoint with three generator repetitions. Its prior repair results used v1 adapters. In the [latest progress comment](https://github.com/jwj1342/TurnItOver/issues/10#issuecomment-6912764875), jwj1342 reports completed v1.1 diagnosis at 2B/4B/8B on Validation and Test, a Linux/Chrome 143 Table 3 replay with an identical CSV, and ongoing v1.1 repair completion. These are collaborator reports; this delivery independently verifies our Validation results and does not run Test or establish the completion of their queued jobs. Both sides use the same admitted 44-case set, but success rates remain separate until the actual commits, stopping, state sampling, Gold and error/retry contracts are aligned. Their proposal to use one harness in the main repair table remains a discussion decision, not an implementation change in this package.

Pending: full Evidence repair, the Qwen3-VL-4B rest/full-scan baselines, v1.1 8B reconciliation, image-removal measurements, qualitative analysis and cost-definition alignment. A further strong generator may be added depending on concurrency and stability; Kimi is not a committed choice. Publishing this package neither starts those experiments nor declares the paper's complete repair table finished.
