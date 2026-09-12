"""Group indistinguishable public prefixes within splits into empirical action targets."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from turnitover.dataset.prepare import SPLITS


def observable_key(record, root, image_hashes):
    protocol, payload_text = record['messages'][0]['content'].rsplit('\n', 1)
    payload = json.loads(payload_text)
    hashes = {}
    for relative in record['images']:
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Image references must be relative paths without parent traversal')
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Image outside source dataset')
        if relative not in image_hashes:
            image_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative] = image_hashes[relative]
    payload['image_order'] = [hashes[p] for p in payload['image_order']]
    for observation in payload['history']:
        observation['image_ref'] = hashes[observation['image_ref']]
    return hashlib.sha256((protocol+'\n'+json.dumps(payload, sort_keys=True)).encode()).hexdigest()


def export_distributions(source, output):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Complete source required')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(version=1, status='incomplete', source=str(source.resolve()),
                  source_manifest_sha256=hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest(),
                  target_semantics='empirical privileged-teacher action distribution conditional on exact public prefix',
                  limitations=['Only exact pixel/context matches are grouped, separately within each split.',
                               'Action frequencies are empirical, not optimal-action probabilities.',
                               'Singleton privileged targets may still be unidentifiable from learner inputs.',
                               'Use per-record loss_weight; ignoring weights changes the grouped objective.'])
    counts, conflicts, image_hashes = {}, [], {}
    try:
        with (output / 'groups.private.jsonl').open('w') as private:
            for split in SPLITS:
                groups = defaultdict(list)
                for line in (source / f'{split}.actions.jsonl').read_text().splitlines():
                    record = json.loads(line)
                    groups[observable_key(record, source, image_hashes)].append(record)
                written = 0
                with (output / f'{split}.actions.jsonl').open('w') as stream:
                    for key, records in groups.items():
                        representative = records[0]
                        targets = Counter(json.dumps(json.loads(r['messages'][1]['content']), sort_keys=True) for r in records)
                        group_id = f'{split}:{key}'
                        group = dict(id=group_id, members=[r['id'] for r in records], targets=dict(targets))
                        private.write(json.dumps(group)+'\n')
                        if len(targets)>1:
                            conflicts.append(dict(id=group_id, examples=len(records), distinct_actions=len(targets)))
                        for relative in representative['images']:
                            path = output / relative
                            path.parent.mkdir(parents=True, exist_ok=True)
                            if not path.exists():
                                shutil.copyfile(source / relative, path)
                        for index, (target, count) in enumerate(sorted(targets.items())):
                            row = dict(id=f'{group_id}:{index}', condition='single_reference_empirical_action_bc',
                                       loss_weight=count/len(records), images=representative['images'],
                                       messages=[representative['messages'][0], dict(role='assistant', content=target)])
                            stream.write(json.dumps(row)+'\n')
                            written += 1
                counts[split] = dict(observable_groups=len(groups), weighted_examples=written)
        report.update(status='complete', counts=counts, conflicts=conflicts,
                      conflicting_groups=len(conflicts), affected_source_examples=sum(r['examples'] for r in conflicts))
    except Exception as exc:
        report.update(error=str(exc))
        raise
    finally:
        (output / 'manifest.json').write_text(json.dumps(report, indent=2))
    return report
