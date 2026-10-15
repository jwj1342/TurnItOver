# Interactive Teacher Code Review

**Date:** 2026-10-15
**Status:** local-only

The requested fresh GPT-6-Astra review was started before real API deployment but hit its usage limit without returning a review. A local blocking review then checked visible-input isolation, bounded retries, frozen framing, objective-truth/evidence separation, export identity/SHA validation, and paid-call caps.

Blocking findings fixed before the final export:

- runtime-only history was incorrectly required to have an image; the audit now checks only image-producing steps;
- the API request cap was checked only outside schema retries; it now bounds every retry call;
- export previously trusted semantic acceptance alone; it now also requires pipeline integrity and a legal terminal verdict;
- evidence audit now rejects annotation/program/label identity mismatch and conservatively declines witness mapping after multiple joints remain actuated.

No remaining blocking issue was found locally. The review is not an independent model approval; the deterministic evidence audit and tests are the acceptance authority for this run.

## Round 3 (joint-axis motion evidence)

The round2 review notes were not available in the repository, so the four requested defects were re-checked locally against the code:

- zero/rest conflation: `next_witness` treated any value-0 detent as rest, and the round2 lookup could therefore bind a `rest` actuation to the wrong witness. Binding now uses actual clamped joint values from a frozen manifest;
- live image to witness binding relied on whichever v4 trajectory happened to visit a witness; every motion-view witness is now re-rendered before any teacher call and cross-checked against the v4 pixels;
- identity: the Gold `id`, Gold/annotation row SHA, manifest SHA and content, and Test absence are now enforced in addition to the round2 fields;
- export: Test/non-formal splits and post-audit Episode edits are blocked; action rows use a label-independent rule, so hidden Gold no longer decides which visible prefixes are supervised.

This is still a local review, not an independent model approval.


## Round 4 (diagnosis bottleneck attribution)

- Evidence standard: Oracle exposure counted 1-px rim shifts (PM_102080) as sufficient evidence. Schema-2 manifests add a perceptibility tier (changed region surviving a 2-px erosion); the audit credits only perceptible exposures when the tier exists. Schema-2 manifests may only add fields; `supersedes_errors` enforces this, and an Episode recorded under a schema-1 manifest is re-audited only through a declared `supersedes` link.
- Hierarchy common input: a rest-pose `query_runtime hierarchy` runs as a declared fixed step in both modes; its payload is checked against the frozen hierarchy in the dry run and in the audit. A teacher-chosen hierarchy query that matches is eligible as action supervision; fixed steps never are.
- Export: repeated conditions on one row produced duplicate row ids; ids are now qualified by the source directory and checked for uniqueness.
- Client: empty provider responses now record `finish_reason` and usage; several round4 empty responses were reasoning-token exhaustion at the 4096 ceiling.
- Browser test `test_catalog_clean_and_mixed_are_shard_independent`: the fixture expectation did not match the seeded draws (no two-corruption sample among idx 0-3). The generator was correct; the fixture now uses 8 samples.

This is still a local review, not an independent model approval.
