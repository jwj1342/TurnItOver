# Hard paired expansion

The released 75-object baseline uses `hard-full-motion-formal-v1`: full native
rest/half/endpoint motion, a 3-degree axis error, a 0.5%-diagonal anchor offset,
or a 2-degree initial part rotation. The older `hard-expanded-v1` bundle uses
`hard-pilot-v2` with different motion intervals and corruption types. Its 75
objects are **not** automatically accepted as equivalent formal candidates.

`hard_tier_expansion.py` preserves the baseline defect artifacts and case IDs,
adds one Clean case per object from the exact rendered normal frames, and
checks the actual normal program/checker SHA binding. Newly generated defects
must pass the unchanged formal collector and source-authenticity gates. Every
case remains pending human review; no training manifest is admitted here.

The target is 150 independent objects, with original 75 Train objects retained.
New objects fill 25 Train, 20 Validation, then 30 Test positions. Category
sampling caps scale with the object denominator; corruption strengths,
visibility gates, native image resolution, and native motion bounds do not.

The collector writes a checkpoint after each attempt. Full `review_manifest.jsonl`
contains both cases for every admitted object, with native media at
`native/{case_id}/{view}/{reference|candidate}_{rest|mid|limit}.png`.
`private/gold.jsonl` and `private/checks.jsonl` contain private construction truth
and real checker evidence. All these artifacts belong in the private dataset
repository. Candidate output and a completed human-reviewed dataset are distinct.

Example current construction invocation (source paths are explicit):

```bash
python scripts/hard_tier_expansion.py \
  --builder-root /path/to/pinned/TurnItOver-hard-pilot \
  --base /path/to/hard-formal-v1 \
  --pool /path/to/hard-candidate-pool-v2/final/pool.json \
  --exclude /path/to/frozen-source-exclusions.json \
  --archive-verification output/additional-official-archive-verification.json \
  --out output/hard-tier-expansion-v1
```

Repeat `--archive-verification` for each separately verified source wave. Each
report is schema-checked and SHA-pinned into the private run state.

The initial run pins its builder checkout in `private/run_state.json`. A changed
input requires a new output directory. The branch also includes the formal
construction dependency closure under `scripts/hard_construction`, reusing
shared render/checker helpers in `scripts/tier_construction`; its provenance
file records the original and packaged module hashes. Import namespaces and
repository-root resolution are the only changes to these vendored helpers.
The initial running job continues using its original pinned checkout.

`verify_hard_archive_provenance.py` is the auditable fallback for source waves that lack their original local acquisition manifest. It queries official Hugging Face LFS metadata at the exact frozen dataset revision, then checks local archive hashes and sizes. Its timestamp is a current verification timestamp, never an invented historical download time.

`hard_tier_expansion_sources.py` screens reserved official conversions with the
original structural policy: at least three joints and five parts, a large
semantically meaningful primary target, native travel and legal endpoint
requirements. It verifies official archive and converted-spec hashes and emits
atomic qualified source records plus explicit rejections. Structural admission
alone never establishes Hard difficulty.

```bash
python scripts/hard_tier_expansion_sources.py \
  --builder-root /path/to/pinned/TurnItOver-hard-pilot \
  --reservations output/medium-hard-reservation-release-v1.json \
  --out output/hard-tier-reserved-sources-v1
```
