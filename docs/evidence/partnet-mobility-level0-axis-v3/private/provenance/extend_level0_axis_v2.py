"""Add genuine independent clean/90-degree-axis pairs to the existing v2 bundle."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from scripts.level0_expansion_search import find_pair
from scripts.level0_expansion_sources import load_pool
from scripts.prepare_level0_axis_review import board, opaque_id, pair_preview
from scripts.prepare_level0_axis_v2 import DEFAULT_OUT, read_rows, seal
from scripts.prepare_obvious_defect_review import DEFAULT_CONFIG, ROOT, STATES, checker_result, sha256_path, write_jsonl
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

DEFAULT_BASE = ROOT / 'data/level0-axis-v2-base-36d32f9'


def run(out=DEFAULT_OUT, base=DEFAULT_BASE, target=28):
    if target < 28:
        raise ValueError('Expansion target must be at least 28 independent objects')
    if not base.exists():
        if out != DEFAULT_OUT or not out.exists():
            raise ValueError('Provide the original 7-object base bundle')
        shutil.copytree(out, base)
    old = json.loads((base / 'manifest.json').read_text())
    for name, digest in old['files'].items():
        if sha256_path(base / name) != digest:
            raise ValueError(f'Base SHA mismatch: {name}')
    original_gold = read_rows(base / 'private/gold.jsonl')
    if len(original_gold) != 14 or len({g['asset_id'] for g in original_gold}) != 7:
        raise ValueError('Expected the reviewed 7-object base snapshot')
    if not out.exists():
        shutil.copytree(base, out)
    elif read_rows(out / 'private/gold.jsonl')[:len(original_gold)] != original_gold:
        raise ValueError('Output does not preserve the reviewed base')
    cfg = load_generate_config(DEFAULT_CONFIG)
    pool, original_splits, provenance = load_pool()
    metadata = {r['spec'].asset_id: r['metadata'] for r in pool}
    public = read_rows(out / 'review_manifest.jsonl')
    gold = read_rows(out / 'private/gold.jsonl')
    checks = read_rows(out / 'private/checks.jsonl')
    reviews = read_rows(out / 'human_review.jsonl')
    original_ids = {g['asset_id'] for g in original_gold}
    selected = {g['asset_id'] for g in gold}
    models = {metadata[a]['model_id'] for a in selected}
    fingerprints = {metadata[a]['original_geometry_fingerprint'] for a in selected}
    checkpoint_path = out / 'private/expansion_screening.json'
    screening = json.loads(checkpoint_path.read_text())['assets'] if checkpoint_path.exists() else []
    attempted = {a['asset_id'] for a in screening}
    (out / 'private/expansion_base_manifest.json').write_text(json.dumps(old, indent=2))
    (out / 'private/source_catalog').mkdir(exist_ok=True)

    def checkpoint():
        write_jsonl(out / 'review_manifest.jsonl', sorted(public, key=lambda r: r['case_id']))
        write_jsonl(out / 'private/gold.jsonl', gold)
        write_jsonl(out / 'private/checks.jsonl', checks)
        write_jsonl(out / 'human_review.jsonl', sorted(reviews, key=lambda r: r['case_id']))
        (out / 'private/expansion_screening.json').write_text(json.dumps({'assets': screening}, indent=2))
        expanded_splits = json.loads(json.dumps(original_splits))
        entries = []
        for aid in sorted(selected):
            clean = next(g for g in gold if g['asset_id'] == aid and g['kind'] == 'clean')
            entry = dict(metadata[aid])
            entry['spec'] = f"../{clean['case_id']}.json"
            entry['spec_sha256'] = sha256_path(out / 'private' / f"{clean['case_id']}.json")
            entries.append(entry)
            expanded_splits['groups'][entry['split_group']] = 'train'
        (out / 'private/source_catalog/catalog.json').write_text(json.dumps({
            'version': 1, 'source_id': 'PartNet-Mobility-Level0-v2-expanded',
            'role': 'level0_train_expansion', 'formal_training': False, 'assets': entries}, indent=2))
        (out / 'private/expanded_splits.json').write_text(json.dumps(expanded_splits, indent=2))

    views = load_views(cfg.views_path)
    with ObservationSession(cfg.render, views) as session:
        for index, candidate in enumerate(pool):
            spec, meta = candidate['spec'], candidate['metadata']
            if len(selected) >= target:
                break
            if spec.asset_id in selected or spec.asset_id in attempted:
                continue
            if meta['model_id'] in models or meta['original_geometry_fingerprint'] in fingerprints:
                screening.append({'asset_id': spec.asset_id, 'status': 'duplicate_source_object'})
                continue
            pair, diagnostics = find_pair(session, ObjectProgram.from_spec(spec), views, 1000000 + index * 50000)
            screening.append({'asset_id': spec.asset_id, 'category': spec.category,
                'original_train': candidate['original_train'],
                'status': 'automatic_pass' if pair else 'rejected', 'joints': diagnostics})
            if pair:
                ref, chosen, joint_id, values = pair
                reference = ObjectProgram.from_spec(spec)
                for kind, seed, program, rendered, label in (
                    ('clean', None, reference, ref, None),
                    ('joint_axis', chosen['seed'], chosen['program'], chosen['render'], chosen['label'])):
                    cid = opaque_id(spec.asset_id, kind, seed)
                    bp, ip = f'boards/{cid}.png', f'inputs/{cid}.png'
                    board(out / bp, cid, ref, rendered, chosen['view'])
                    pair_preview(out / ip, ref, rendered, chosen['view'])
                    result = checker_result(program, rendered, reference, ref)
                    if result['code_correct'] != (kind == 'clean'):
                        raise ValueError(f'Checker disagreement: {cid}')
                    public.append({'case_id': cid, 'category': spec.category, 'view': chosen['view'],
                        'states': list(STATES), 'input_states': ['rest', 'limit'], 'image_size': [256, 256],
                        'board': bp, 'input_preview': ip, 'status': 'pending_human_review'})
                    gold.append({'case_id': cid, 'asset_id': spec.asset_id, 'kind': kind,
                        'expected_status': 'pass' if label is None else 'fail',
                        'defect_id': None if label is None else label.defect_id,
                        'joint_id': None if label is None else joint_id,
                        'parts': [] if label is None else list(label.parts),
                        'parameters': {} if label is None else label.params,
                        'candidate_sha': program.sha, 'reference_sha': reference.sha,
                        'training_eligibility': 'pending_human_review'})
                    checks.append({'case_id': cid, **result})
                    reviews.append({'case_id': cid, 'status': 'pending_human_review', 'reviewer': None,
                        'reviewed_at': None, 'human_judgment': {'verdict': None, 'defect_id': None,
                            'one_glance': None, 'normal_vs_candidate_motion': None, 'evidence_states': [],
                            'notes': None, 'rest_matches_reference': None, 'no_clipping_or_occlusion': None}})
                    (out / 'private' / f'{cid}.json').write_text(json.dumps(to_dict(program.spec)))
                selected.add(spec.asset_id)
                models.add(meta['model_id'])
                fingerprints.add(meta['original_geometry_fingerprint'])
            checkpoint()
            print(json.dumps({'asset_id': spec.asset_id, 'automatic_pass': pair is not None,
                'independent_objects': len(selected), 'target': target}), flush=True)
            if len(selected) >= target:
                break
    checkpoint()
    # Preserve every original case's actual human judgment and native image bytes.
    current_reviews = {r['case_id']: r for r in reviews}
    assert all(current_reviews[r['case_id']] == r for r in read_rows(base / 'human_review.jsonl'))
    for name, digest in old['files'].items():
        if name.startswith(('boards/', 'inputs/', 'frames/')) or (name.startswith('private/l0-') and name.endswith('.json')):
            assert sha256_path(out / name) == digest, name
    inputs = dict(old['inputs'])
    inputs.update(provenance)
    inputs.update({str(p.resolve()): sha256_path(p) for p in
        (Path(__file__), ROOT / 'scripts/level0_expansion_search.py', ROOT / 'scripts/level0_expansion_sources.py')})
    seal(out, inputs)
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest.update(source_catalog='private/source_catalog/catalog.json', expanded_splits='private/expanded_splits.json',
        target_objects=target, base_objects=7, base_manifest_sha256=sha256_path(base / 'manifest.json'),
        base_commit='36d32f9', expansion_protocol='native-endpoint-motion; normal-only frozen padding; unchanged pixel gates')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    from scripts.finalize_level0_axis_v2 import finalize
    return finalize(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--target', type=int, default=28)
    args = parser.parse_args()
    print(json.dumps(run(args.out.resolve(), args.base.resolve(), args.target)))
