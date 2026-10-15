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
