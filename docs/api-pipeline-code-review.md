# API pipeline preflight code review

Review status: **PROVISIONAL GO for one frozen real initial generation**, followed by the agreed artifact checks before interpreting paired diagnosis scores. No remaining code blocker was identified for that initial request. This is a same-family Codex subagent review, not an independent model-family audit and not evidence that the paid pipeline has succeeded.

Reviewed branch: `laq/api-pipeline`; HEAD: `99db3a767ed78b5c533ad5c4bbc88c8fb09bb46d`; working tree includes uncommitted implementation. Reviewer performed no API calls and changed no implementation files. The parent agent applied fixes while this review was in progress; final positions below refer to the files reread after those fixes.

## Blocking findings and disposition

1. **Resolved — contradictory final diagnosis schema.** `turnitover/verifier/final_prompt.txt:10` initially required nine fields while the example and `turnitover/verifier/contracts.py:129` require eight. This could cause rejected diagnoses and extra paid schema-correction attempts. The prompt now says eight and matches the parser.
2. **Resolved — scorer could accept an unrelated prepared bundle.** `turnitover/api_pipeline/scoring.py:122` now compares the prepared manifest SHA with the paired run and rechecks all frozen input hashes before emitting scores. The initial version accepted an arbitrary `--prepared` path.
3. **Resolved by explicit scope restriction — material defects were mislabeled as clean.** `quality()` reuses `validate_code()`, which checks contract, geometry and runtime budgets, not appearance. A real local Chromium counterexample changed the drawer from blue to black and still returned `status=complete`, `code_correct=True`, and invariant pass fraction `1.0`. `turnitover/api_pipeline/fixture.py:22` now excludes material diagnosis from scope; `scoring.py:58` separates those findings and avoids charging a material-only clean-case failure as a detection error. “Correct” must retain this declared scope.
4. **Mitigated, with actual generated-artifact inspection still necessary — injections depend on the generated scene graph.** `fixture.py:58` mutates a local drawer position and `fixture.py:63` replaces the named pivot quaternion setter. A geometrically valid generated implementation may use different local frames or an unused named pivot. `scoring.py:94` now checks that injected errors affect only the intended part and have the expected rest/motion visibility pattern; failed checks suppress diagnosis scoring. This catches extra/rest-pose faults, but is not a proof that every remaining motion mismatch is specifically a wrong axis. Inspect the actual generated code and private per-state geometry before describing the controlled fault as semantically verified. Do not resample a failed base or infer label validity from the synthetic-reference unit fixture alone.

## Verified interfaces and isolation

- **No CheckResult key-name bug:** `turnitover/core/sample.py:38` defines `invariants`; `Invariant` defines `passed` and `parts`. The accesses in `scoring.py:23` and `scoring.py:49` match the serialized dataclasses. `Evidence` receives valid state-to-part geometry mappings.
- `paired.py` freezes one generated source, one reference PNG and task, and verifies source/image hashes. Both arms receive the same initial program. Separate iterative calls create fresh browser/verifier instances and use the same `FrozenAPI` for generator, verifier and repairs.
- Both arms use `exact_budget=True`; fixed collection is recorded before final inference. Both use the same final prompt after K observations. Active policy's extra action-selection calls remain explicit cost, not extra observations.
- The public model prompt receives the authored normative task, public reference images and acquired observations. No private reference program, target labels, privileged validation result, or future observations is passed through `run_iterative()` to a model. Repair receives candidate source and evidence-linked public feedback.
- `FrozenAPI` records attempts before sending, copies and hashes request images, records raw request/response and usage, freezes the returned model identity, and stops after transport/model identity failures. Request/trace handling does not serialize configured endpoint URLs or authentication headers.
- Fresh verifier execution follows each successful repair/render round. Independent validation is post-hoc and does not control verifier decisions or repair content.

## Nonblocking limitations to retain in the run report

- **Validation retries remain enabled.** `turnitover/verifier/loop.py:16` and `:45` permit three total attempts for one invalid decision, including an early verdict in exact-budget mode. These are recorded paid schema corrections, distinct from transport retries, which are disabled. “No retries” without this distinction would be false.
- **The fixed K=4 schedule cannot observe the second joint.** `turnitover/verifier/policies.py:21` yields stats, front, oblique_fr, then drawer limit. Therefore the door-axis case deliberately tests an observation coverage difference. One authored object and one sample per arm support a pilot trace, not general superiority or a fair equal-token claim.
- **Geometric witness score is only a proxy.** `scoring.py:29` compares geometry after replay and requires an acquired image. It does not prove image visibility, natural-language explanation correctness, or whether every cited step supports the finding. The output already states this limit.
- **Correctness is finite sampled-state correctness.** `gt_validator.py:25` evaluates rest and each joint independently at mid/limit; it does not exhaust combined states or arbitrary controls.
- **Request count is not a monetary cap.** `api.py:61` accurately records `dollar_cap_enforced=False`; separate prepare/run API ledgers have separate count limits. Do not report the limit as a provider-enforced spending ceiling or a shared total across stages.
- The prepared manifest compares public configuration and returned model ID, but it does not attest provider backend weights or endpoint identity across different processes. This is acceptable for the frozen local configuration pilot; it is not a general reproducibility guarantee.

## Reviewer validation

- Executed offline focused tests: `tests/unit/test_api_recording.py` and `tests/unit/test_verifier_exact_budget.py`; **15 passed**. All mocked transport calls stayed local.
- Executed the real Chromium wrong-color counterexample above using the compatible TurnItOver `.venv-local` and its pinned `.cache/ms-playwright`. An initial attempt with the unrelated home browser cache failed before browser launch; rerun used the pinned cache successfully. No model/API request was involved.
- Read the new browser fixture test and scorer tests. Parent-reported wider test totals are not claimed as reviewer-executed results.
- Read all requested implementation modules and the reused iterative repair, verifier contracts/loop, browser harness, evidence collector and checker schemas.

Final decision: **No unresolved blocker for one real `prepare` request.** Paired outcome claims remain conditional on the actual generated base/injection checks and completed trace artifacts; this review does not authorize a model swap, retry after a failed gate, or a broader experiment.
