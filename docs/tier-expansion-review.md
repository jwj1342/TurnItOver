# Full human review for the tier expansion

The owner reviews all candidate cases. Automatic acceptance only produces
`pending_human_review`; pending, rejected, missing, or contradictory judgments
never count as qualified cases. Historical spot checks do not replace this review.

`prepare_tier_review.py` freezes an answer-free offline pack. Each case has an
opaque ID, original 256px evidence, and optional supplementary states/views.
The private custody mapping is written outside the reviewer directory. The
review page offers Easy, Medium, Hard, Clean, and invalid decisions without
showing expected labels or source identities. It stores work in the browser
under the package's case IDs and reviewer identity; the reviewer explicitly
exports a JSONL return. Each return binds the actual package SHA256. The program
does not authenticate human identity or turn pending templates into approvals.

```sh
python -m scripts.prepare_tier_review build \
  --intake /private/intake.json --output /private/new-review-pack \
  --custody /private/new-review-custody --namespace unique-wave-name
python -m scripts.prepare_tier_review validate --root /private/new-review-pack
```

Intake is a JSON list. Each row has `source_case_id`, `bundle`, and `primary` /
`supplementary` image lists, each containing `path` and `caption`. Captions are
restricted to reference/candidate, initial/supplementary view and state. No
Gold, source identity, expected difficulty, previous response, or mapping is
copied into the reviewer pack. New directories are required. Exact media bytes
are SHA256-bound; changed files or extra files fail validation.

For Easy batches, `tier_review_intake.py` uses the four native rest/limit images
as primary evidence and extracts the original lossless 256px mid tiles from the
existing three-state board. It excludes the board's case-ID header. This does
not synthesize a frame or change its pixel values.

Return custody must preserve the original exported file. Any later normalization
for aggregate admission must bind that file, its package, the opaque mapping,
and the case evidence. Reviewer disagreement is retained. Rejected objects are
replaced by newly constructed sources with the same gates; existing split
membership and evaluated v2 remain fixed.
