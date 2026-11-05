# Hard full-motion formal v1.1

`hard-full-motion-formal-v1.1` preserves the immutable 75-object
`single_motion_v1` baseline and adds 75 independently sourced
`compositional_v1_1` objects.  The two tracks are recorded per case and must
be reported separately.

Each compositional case has exactly two declared field faults on one moving
part.  Supported signatures are the six pairs of `joint_axis`,
`joint_anchor`, `part_offset`, and `part_rotation`.  Each fault has an
Easy/Medium-range parameter; Hard difficulty is the need to diagnose both,
not a large-motion source gate.  The combined candidate must change exactly
the two declared fields, retain source authenticity, and have complete normal
and candidate checker/render evidence.  Individual atom screening is not an
admission requirement.

The candidate bundle has 150 source-unique objects and 300 cases (one Clean
control and one defect candidate per object), split 100/20/30 for
Train/Validation/Test.  All cases remain `pending_human_review` and the train
manifest remains empty until review and a separate eligibility freeze.

## Private three-tier SFT candidate

The experiment owner has authorized a separate, private programmatic
candidate for the first three-tier SFT run.  It is published in the access
controlled `mmk1209/TIO-datasets` branch
`laq/three-tier-900-sft-v1-candidate` at `68ee70bc09fa5b5f47a6053c9059ee543b7eca2b`.
The public repository carries no private RGB, AssetSpec, or answer artifact.

`releases/three-tier-900-sft-v1-candidate/` contains the final 900 Episode
export; the Easy/Medium input packages; a global source-split ledger; and a
compact Hard roster sidecar that binds all 150 Hard source identities, spec
hashes, and frozen splits to the pre-existing Hard v1.1 private candidate
archive.  The sealed bundle SHA256 is
`185d2d6ba7ca2b6d2fb7426d407e50c9682a8632e3a827f18502cab9c309e2e0`.

The split is 200/40/60 cases per difficulty for Train/Validation/Test
(600/120/180 total).  Its release status is `programmatic_candidate` with
human review explicitly deferred by the experiment owner: it is authorized
for the requested experimental SFT work, but is not a final benchmark release
or evidence of human acceptance.  The accompanying exporter implementation
is published separately on `laq/three-tier-900-sft-export`.
