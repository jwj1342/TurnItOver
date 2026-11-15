"""Derive the withheld-evidence validation export from the three-tier-900 episode export.

For each defective validation case, remove every observation cell (reference and candidate image of that view and
joint state) that the evidence-constrained target cites for any gold label. The first reference image (the task
reference, image_order[0]) is always kept because it is never a candidate image and cannot expose a defect on its own.
Cases are kept only if at least one cell was removed and at least one non-exposing cell remains. The user prompt is
rebuilt byte-exactly in her format with the reduced history/image_order and remaining=0. The assistant target becomes
an uncertain verdict with no findings (no remaining step can be cited), and evidence.covered_label_indices becomes [].
Images are copied (not symlinked: her loader resolves paths and rejects anything outside the dataset root).
Usage: python analysis/build_withheld_export.py
"""
from __future__ import annotations
import os
import hashlib, json, shutil
from collections import Counter, defaultdict
from pathlib import Path
TIO = Path(os.environ.get('TIO_RUNS', '/scratch/jwj/tio-runs'))
SRC = TIO / 'data900/three-tier-900-sft-v1/episode_export'
DST = TIO / 'data900/three-tier-900-withheld-v1/episode_export'
INV = TIO / 'action900/inventory/case-inventory.private.jsonl'


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    if DST.exists():
        raise SystemExit(f'refusing to overwrite {DST}')
    inv = {r['id']: r for r in rows(INV) if r['split'] == 'validation'}
    ann = {a['id']: a for a in rows(SRC / 'annotations.private.jsonl') if a['split'] == 'validation'}
    stats = Counter(); removed = Counter(); kept_cells = Counter(); by_type = Counter()
    out_records, kept_ids = [], []
    for ep in rows(SRC / 'validation.episodes.jsonl'):
        cid = ep['episode_id']; a = ann[cid]; r = inv[cid]
        if not a['labels']:
            stats[('skip_clean', ep['difficulty'])] += 1; continue
        content = ep['messages'][0]['content']; i = content.index('{"context"')
        prefix, payload = content[:i], json.loads(content[i:])
        assert prefix + json.dumps(payload) == content, cid
        step_img = {h['step']: h['image_ref'] for h in payload['history']}
        target = json.loads(ep['messages'][-1]['content'])['verdict']
        cited = {step_img[s] for f in target['findings'] for s in f['evidence_steps']}
        cell_of = {}
        for cell in r['observation_cells']:
            key = (cell['view_id'], cell['state'])
            cell_of[cell['candidate_image']] = key; cell_of[cell['reference_image']] = key
        exposing = {cell_of[img] for img in cited}
        all_cells = [(c['view_id'], c['state']) for c in r['observation_cells']]
        keep = [c for c in r['observation_cells'] if (c['view_id'], c['state']) not in exposing]
        if not exposing:
            stats[('skip_nothing_exposed', ep['difficulty'])] += 1; continue
        if not keep:
            stats[('skip_no_cell_left', ep['difficulty'])] += 1
            for lab in a['labels']: by_type[(lab['defect_id'], 'excluded')] += 1
            continue
        keep_imgs = {payload['image_order'][0]} | {p for c in keep for p in (c['reference_image'], c['candidate_image'])}
        images = [p for p in payload['image_order'] if p in keep_imgs]
        sha = dict(zip(ep['images'], ep['image_sha256']))
        history = [{**h, 'step': n} for n, h in enumerate(h for h in payload['history'] if h['image_ref'] in keep_imgs)]
        assert images[1:] == [h['image_ref'] for h in history], cid
        new_payload = dict(payload); new_payload['history'] = history; new_payload['image_order'] = images; new_payload['remaining'] = 0
        verdict = {'verdict': {'status': 'uncertain', 'confidence': 0.0, 'findings': [],
                               'summary': 'The remaining observations do not expose a defect, but do not establish that the candidate is clean.',
                               'limitations': ['Observations that expose the annotated defects were withheld.']}}
        rec = dict(ep)
        rec['images'] = images; rec['image_sha256'] = [sha[p] for p in images]
        rec['messages'] = [{'role': 'user', 'content': prefix + json.dumps(new_payload)}, {'role': 'assistant', 'content': json.dumps(verdict)}]
        rec['evidence'] = {**ep.get('evidence', {}), 'covered_label_indices': [], 'uncovered_label_indices': list(range(len(a['labels']))), 'status': 'withheld'}
        rec['withheld'] = {'removed_cells': sorted(map(list, exposing)), 'kept_cells': [[c['view_id'], c['state']] for c in keep]}
        out_records.append(rec); kept_ids.append(cid)
        stats[('kept', ep['difficulty'])] += 1; removed[ep['difficulty']] += len(exposing); kept_cells[ep['difficulty']] += len(keep)
        for lab in a['labels']: by_type[(lab['defect_id'], 'kept')] += 1
        for p in images:
            (DST / p).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(SRC / p, DST / p)
    (DST / 'validation.episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in out_records))
    (DST / 'annotations.private.jsonl').write_text(''.join(json.dumps(ann[c]) + '\n' for c in kept_ids))
    man = json.loads((SRC / 'manifest.json').read_text())
    tiers = Counter(r['difficulty'] for r in out_records)
    man['counts'] = {'validation': {'difficulty': dict(tiers), 'episodes': len(out_records), 'stopping': 0}, 'train': {'episodes': 0, 'stopping': 0}, 'test': {'episodes': 0, 'stopping': 0}}
    man['dataset_version'] = 'three-tier-900-withheld-v1'
    man['derived_from'] = {'dataset_version': 'three-tier-900-sft-v1', 'manifest_sha256': hashlib.sha256((SRC / 'manifest.json').read_bytes()).hexdigest(),
                           'release_zip_sha256': '185d2d6ba7ca2b6d2fb7426d407e50c9682a8632e3a827f18502cab9c309e2e0'}
    man['withheld_rule'] = ('defective validation cases only; every cell cited by the evidence-constrained target for any gold label removed; '
                            'first reference image kept; cases kept only if something was removed and a non-exposing cell remains; targets set to uncertain; built by analysis/build_withheld_export.py')
    (DST / 'manifest.json').write_text(json.dumps(man, indent=2, sort_keys=True) + '\n')
    report = {'stats': {f'{k[0]}:{k[1]}': v for k, v in sorted(stats.items())}, 'removed_cells_per_tier': dict(removed), 'kept_cells_per_tier': dict(kept_cells),
              'labels_by_type': {f'{k[0]}:{k[1]}': v for k, v in sorted(by_type.items())}, 'kept_ids': kept_ids}
    (DST / 'PROVENANCE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'kept_ids'}, indent=1))
    print('manifest sha256', hashlib.sha256((DST / 'manifest.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
