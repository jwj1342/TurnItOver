# Release validation — 2026-11-16

Scope: the two reproduction packages added relative to pinned runtime commit `89e07db82890af02ae651dcd8b86082981fb7d1b`. Checks below use private inputs locally; only aggregate validation evidence is published.

## Code, inputs and behavior

- Repository unit checks: 229 passed, 5 skipped (Torch-dependent checks). This includes the two original research-snapshot byte checks, also run separately. The new package's final behavioral suite has **16 passed, 0 skipped**: frozen identities, input boundaries, source-view budgets, image/feedback isolation, submission budgets, public pass versus independent success, invalid audits, unknown physical findings, input leakage, and refusal to rerun incomplete attempts automatically.
- All 35 Python files in the combined delivery parse successfully. The 20 frozen source/renderer files match their archived raw SHA-256 values; 115 runtime files match the pinned base after LF normalization. Frozen CRLF bytes are intentionally retained. Git whitespace checking with `cr-at-eol` passes; converting those files to LF would invalidate their identities.
- Fresh preparation verified the v1.1 manifest, bound train/validation records and annotations, all training input image hashes, the 44 validation case/source-object identities, frozen images, source programs/specs and renderer hashes. Existing output directories are rejected. The final preparation-only check also passed after adding the training-image verification.
- `npm ci --ignore-scripts` and `npm run build` succeeded. Both generated JavaScript renderer files match SOAR. The archived minimal HTML entry is supplied separately and must replace the default entry as described in README; it matches the third renderer hash.

## Fresh no-model preflight

- **44/44** cases passed fresh browser replay, reference positive controls and initial-candidate failure checks. All **286** initial reference/candidate images match the frozen inputs pixel for pixel. No cases were removed and no pixel or geometry tolerance was relaxed.
- Pixel calibration used Train only: 600 records, 579 pass/fail targets (300/279), 21 uncertain excluded. Threshold **0**, training balanced accuracy **1.0**, matching the frozen rule.
- Final v4 Gold independently passed all **44** initial/positive-control pairs. Reconstructed all **54** original injection groups: **9 rest-visible, 42 motion-visible, 3 unexposed**. The initially equivalent anchor remains in the denominator without restoration credit.
- Windows local environment: Python 3.12.8, Playwright 1.57.0, Chromium headless 143.0.7499.4/build 1200, NumPy 2.4.2, OpenHands SDK 1.62.0. SOAR's observed repair environment is Python 3.12.3 / Playwright 1.58.0; its direct dependencies are recorded in `environment.json`. This is an observed successful replay across these environments, not a guarantee for arbitrary browsers or hardware.

## Recorded repair results independently rechecked

- Reaggregated the private archive for all **660** completed trajectories. Every field in `analysis.json`, including success counts, confidence intervals, per-injection bounds and costs, is identical to the original final analysis. Success counts remain **2 / 5 / 37 / 6 / 71**, each out of 132.
- Independently rescored the final source of every trajectory using the frozen v4 physical predicates: **660 source hashes and 810 original-injection outcomes checked**. All discrete outcomes match. Cross-platform numeric fields use `atol=1e-12, rtol=1e-9`; maximum observed difference was **5.55e-17**, across **10** fields with roundoff. This numeric comparison is not a change to the scoring predicates or thresholds.
- This recheck did not rerender all 660 repaired programs. Their recorded geometry/runtime audits and the original input audit of 660 trajectories remain the source for final geometric success. The fresh replay above covers the full initial 44-case set and Gold controls. Geometric success and per-injection physical restoration remain separate measurements.

## Table 2/3 evidence retained

The Table 2/3 package is unchanged from `3ca533a72f972cadc82da19e28deead94865bdcc`. Its [same-day validation record](../table23-reproduction/VALIDATION.md) documents independent verification of **42 groups / 3,332 existing predictions** and a fresh complete Table 3 replay: **34 cases, 1,340 images, 13,872 configurations, 6,733 distinct trajectories and 32,640 actions**, with image/results/bootstrap/CSV equality. Those checks were completed in the earlier delivery phase and are not claimed as a second run here. jwj1342 separately [reports a matching Linux/Chrome 143 Table 3 CSV](https://github.com/jwj1342/TurnItOver/issues/10#issuecomment-6912764875); that report is distinguished from our local validation.

## Publication boundary

All **53** added/changed files are allowlisted to these two packages and checked as staged Git blobs. The checks cover known workspace credentials without printing them, common API token/private-key patterns, private server paths, all 900 private case IDs, embedded images, unexpected binaries and oversized files. No match was found. Frozen code and the public documentation were reviewed; datasets, annotations, case programs/images, adapters, event logs, browser profiles and login scripts are excluded. The check covers this delivery, not the base repository's entire history.

No training, GPU inference, paid API generation or Test evaluation was rerun to publish the package. No new model result is claimed. The GPU service keeps the original serving/tokenization/generation code and adapts model/adapter paths; this portable service wrapper was not executed on a GPU during release validation. New API trajectories cannot be expected to equal archived stochastic outputs. Missing experiments, parser/state-sampling limitations and unresolved cross-harness choices are described in [RESULTS.md](RESULTS.md).
