# Three-tier expansion contract

The expansion contract in `benchmark/releases/tier-expansion-v1.json` keeps the
current Easy `rc.2`, Medium `rc.1`, and Hard `rc.1` packages as immutable
baselines. It freezes their object/case inventory digests and the 19 source
identities already shared by Easy and Medium. Build the expansion into a new
private release directory; do not update any baseline package or evaluated v2
artifact in place.

The private manifest also records the source identities from the evaluated
Level 0 v2 object list. Its count and sorted-set digest are pinned by the
contract. Existing baseline rows may retain those historical identities, but
new objects and added cases on baseline objects may not reuse them or their pinned geometry.

The Train, Validation, and Test values are minimum counts. Existing object
assignments remain fixed, so Easy retains its 117 Train objects even though the
Train minimum is 100. A baseline object with no prior split records
`original_split: null`; its new split assignment is recorded separately.

The private expansion inventory has this shape:

```json
{
  "version": 1,
  "frozen_evaluation_source_identities": ["PartNet-Mobility:<model-id>", "..."],
  "objects": [
    {
      "tier": "easy",
      "source_identity": "PartNet-Mobility:<model-id>",
      "geometry_fingerprint": "<sha256>",
      "membership": "new",
      "original_split": null,
      "split": "test",
      "cases": [
        {
          "case_id": "<opaque-id>",
          "kind": "clean",
          "defect_id": null,
          "membership": "added",
          "automatic_status": "passed",
          "automatic_evidence_sha256": "<sha256>",
          "difficulty_review": {
            "status": "passed",
            "reviewer": "<reviewer-id>",
            "reviewed_at": "<date-or-timestamp>",
            "verdict": "clean",
            "difficulty": "clean"
          }
        },
        {
          "case_id": "<opaque-id>",
          "kind": "defect",
          "defect_id": "kinematics.joint_axis",
          "membership": "added",
          "automatic_status": "passed",
          "automatic_evidence_sha256": "<sha256>",
          "difficulty_review": {
            "status": "passed",
            "reviewer": "<reviewer-id>",
            "reviewed_at": "<date-or-timestamp>",
            "verdict": "defect",
            "difficulty": "easy"
          }
        }
      ]
    }
  ]
}
```

Each object must have at least one Clean and one defect case. Every added case,
including controls added to a baseline object, needs a passing automatic
evidence hash and a human review record whose verdict and difficulty match the
case. New objects must be globally unique by source identity and geometry.
Only the 19 frozen Easy/Medium baseline overlaps are allowed. A rejected or
pending case does not count toward any quota.

Run the aggregate gate after assembling the private inventory:

```sh
python scripts/validate_tier_expansion.py \
  --manifest data/TIO-datasets/releases/tier-expansion-v1/expansion-manifest.json \
  --contract benchmark/releases/tier-expansion-v1.json \
  --output output/tier-expansion-v1-validation.json
```

The report is a count and gate summary and does not include source identities,
Gold labels, or reviewer notes. The contract is only a candidate gate; passing
it does not freeze the benchmark or make the data training-ready.

## Required local evidence bindings

An added case must include `evidence` with `automatic_report`, `review_package`,
and `review_return` references. Each reference has `path` (relative to the private
expansion manifest directory) and `sha256`. Absolute paths, parent traversal,
escaping symlinks, absent files and changed bytes fail validation. The existing
`automatic_evidence_sha256` must equal the automatic report reference hash.

All three documents are JSON objects with `version: 1`. The automatic report
has `status: passed`, a `binding` object containing exactly `case_id`,
`source_identity`, `geometry_fingerprint`, `tier`, `kind`, and `defect_id`, and a
nonempty `media` list of path/SHA references to the actual checked image files.
The answer-free review package inventory has `case_id` and the identical `media`
list. The returned judgment has `case_id`, `review_package_sha256`, and
`difficulty_review` identical to the admitted case record. Retain the actual
return document from the reviewer; the validator never generates approvals.
An importer for an existing return format must retain the original return and
its provenance; translating a pending template is not approval.

These checks establish local file and case bindings. They cannot authenticate
a human identity, establish that a person actually inspected images, rerun the
geometry checker, or prove that a claimed report was produced by the recorded
checker. Human return custody and independent automatic replay remain required
release responsibilities. The validator's report states this limited scope.

## Trusted custody and geometry pins

`build_tier_expansion_pins.py` reads the original private baseline data ZIPs,
checks each ZIP SHA against the public historical release contract, and seals
all ZIP member bytes. It also seals the entire actual evaluated Level 0 v2
directory (including exact file inventory), its source catalog, the conversion
algorithm source, and existing conversion/qualification records. It writes a
new private JSON file; it never changes source artifacts. Only this private
file's SHA is included in the public expansion contract.

Each private expansion manifest needs `custody_pins: {"path": "...",
"sha256": "..."}` referring to that exact file. Paths inside this custody
inventory locate originals on the custodian machine; it is a custody check,
not a portable release artifact. Preserve a relocation policy and all original
bytes before moving custody. The current sealed file is
`output/tier-expansion-custody-v4/custody-pins.json` (1358 source identities,
28 evaluated v2 objects, 610 protected files). It includes acquisition waves
4-6 and the official LFS verification report for legacy Hard archives. No
baseline identity or geometry intersects evaluated v2 in this sealed inventory.

Every new object must also have `original_spec: {"path": "...",
"sha256": "..."}` relative to its private manifest directory. Its bytes must
match the trusted conversion's original spec SHA, and its source identity and
geometry fingerprint must match that same conversion record. The original
spec and official archive bytes recorded in custody are checked again on
admission. Geometry uses the existing conversion scale audit: unique world
vertices at rest rounded to six decimals, lexicographically sorted float64
bytes and SHA256; the source algorithm is itself pinned. This is a trusted
conversion mapping check, not a claim that aggregate validation recomputes
meshes or reruns checker/rendering. A fresh conversion needs a fresh reviewed
pin inventory and public digest; a self-declared new fingerprint is rejected.

An added case on either a new or baseline object is rejected for an evaluated
v2 identity or canonical geometry match. Baseline object fingerprints must
also equal their pinned source bindings. The prior baseline inventory digest
continues to protect split and case membership. The original v2 directory,
baseline ZIPs and all ZIP member bytes are reread at validation, so their
mutation cannot be hidden by rewriting the expansion manifest.

Each acquisition wave is supplied explicitly and may be repeated:

```sh
python scripts/build_tier_expansion_pins.py \
  --workspace /path/to/ThreejsJudge \
  --catalog-root /path/to/wave-a/catalog \
  --download-manifest /path/to/wave-a/download-manifest.json \
  --catalog-root /path/to/wave-b/catalog \
  --download-manifest /path/to/wave-b/download-manifest.json \
  --output output/new-custody-version/custody-pins.json
```

Existing dataset custody catalogs and Hard qualified-source records remain
implicit workspace inputs; additional waves are never inferred from a name.
Explicit missing sources fail. The builder also pins the converter's assets
and core Python dependencies and the official Hard download manifests. After
all waves and code inputs are stable, construct a new inventory and update the
public contract once to that final digest; old inventories remain unchanged.

All occurrences of one source identity must share the same Train/Validation/Test
split, including the allowed 19 Easy/Medium baseline overlaps. Historical reuse
permission never permits cross-split leakage.
