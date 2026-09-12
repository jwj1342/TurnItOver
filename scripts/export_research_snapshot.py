"""Archive selected measurement records, excluding assets, prompts, images and credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


SUMMARIES = (
    'data/benchmarks/replicacad-v1/benchmark.json',
    'data/benchmarks/replicacad-v1/splits.json',
    'data/benchmarks/oracle-v1/manifest.json',
    'data/benchmarks/paired-v1/manifest.json',
    'data/benchmarks/oracle-action-distributions-v1/manifest.json',
    'output/replicacad-fk-v3/audit.json',
    'output/oracle-observable-ambiguity-v1.json',
    'output/qwen-evidence-v1/metrics.json',
    'output/qwen-evidence-final-v1/metrics.json',
    'output/qwen-active-benchmark-v1/metrics.json',
    'output/qwen-pair-readout-v1/metrics.json',
    'output/qwen-defect-readout-v1/metrics.json',
    'output/repair-generator-sol-smoke-v1/manifest.json',
    'output/repair-generator-sol-smoke-v1/audit.private.json',
    'output/repair-pilot-prepared-v2/manifest.json',
)
REPAIR_PATTERNS = (
    'summary.json', 'analysis.json', '*/*/initial.audit.private.json', '*/*/result.json',
    '*/*/round-*/model.json', '*/*/round-*/audit.private.json', '*/*/round-*/response.txt',
)


def export(root: Path, out: Path):
    sources = [root / name for name in SUMMARIES]
    repair = root / 'output/repair-sol-pilot-v1'
    for pattern in REPAIR_PATTERNS:
        matches = sorted(repair.glob(pattern))
        if not matches:
            raise ValueError(f'Missing measurement records: {pattern}')
        sources.extend(matches)
    if any(not p.is_file() for p in sources):
        raise ValueError('Required source measurements are missing')
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for source in sources:
        relative = source.relative_to(root)
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[str(relative)] = dict(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                    bytes=target.stat().st_size)
    manifest = dict(version=1, scope='measurement_snapshot_not_full_replay_bundle', files=files,
        limitations=['No images, full programs, assets, weights or API credentials included.',
                     'Historical paths in copied records refer to the original workspace.',
                     'Source hashes support traceability; omitted source files are not recoverable from hashes.',
                     'Historical limitations describe each run at its execution time; see the current research status.'])
    (out / 'snapshot.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return dict(files=len(files), bytes=sum(f['bytes'] for f in files.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.out)))
