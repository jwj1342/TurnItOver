# Experiment Audit Report

**Date**: 2026-10-14
**Auditor**: fresh same-family Codex reviewer (read-only, provisional)
**Scope**: frozen v3 supervision, saved first-round checkpoints and Validation outputs

## Overall Verdict: FAIL

## Checks

### A. Ground Truth Provenance: FAIL

The corruption Gold and counterfactual exposure records are independent private audit data, and the saved model inputs do not contain those fields. However, 37/110 Train and 10/28 Validation v3 terminal assistant targets describe injected or otherwise private state. The old evidence checkpoint therefore learned from targets whose explanations were not functions of the visible prefix alone.

### B. Score Normalization: PASS

Status, exact verdict, finding precision/recall/F1 and the asset-cluster bootstrap use raw counts. No metric is normalized by the evaluated model's own maximum, mean or output distribution.

### C. Result File Existence: PASS

The controlled comparison and diagnostic files exist and reproduce the reported terminal counts. The three saved model conditions each contain 28 terminal predictions; the causal-prefix evaluations each contain 110 rows.

### D. Dead Code And Metric Semantics: FAIL

The old terminal report exposed `stop_accuracy` and `final_verdict_rate` as aliases of legal terminal JSON output, although terminal-only evaluation does not measure stop timing. This makes the old stopping claim invalid even though the underlying predictions are real.

### E. Scope Assessment: FAIL

The evaluation contains 28 episodes from seven independent Validation objects, one training seed, two synthetic defect types, no pass-qualified example and no natural-error example. Terminal status is perfectly associated with stage metadata. Claims of learned visual evidence use or learned stopping are unsupported at this scope.

### F. Evaluation Type: simulation_only with independent private Gold

The evaluation uses deterministic corruption Gold and counterfactual pixel-exposure audits in a rendered simulator. These are not model-generated references, but exposure is a proxy for recognizability and cannot be presented as human or VLM evidence recognition.

## Action Items

- Rewrite assistant explanations so equivalent visible prefixes cannot vary with private Gold.
- Export explicit per-prefix `continue/stop` targets under one named stopping policy.
- Separate `valid_output_rate`, diagnostic accuracy and first-stop timing.
- Add a no-image stage-metadata baseline and same-stage evidence-sufficient/insufficient comparisons.
- Retrain before making any claim about the repaired supervision.

## Claim Impact

- Saved first-round metrics: reproducible descriptive results.
- Evidence SFT reduces unsupported findings versus ordinary SFT: supported as a paired old-checkpoint observation.
- Evidence SFT learns visual evidence: unsupported.
- Either SFT learns correct stopping: unsupported.
- Repaired v4 supervision improves a trained model: not evaluated; no retraining was performed.

This audit predates the completed v4 remediation. It does not certify a repaired-training result.
