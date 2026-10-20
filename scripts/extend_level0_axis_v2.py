"""Add genuine independent clean/90-degree-axis pairs to the existing v2 bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from scripts.level0_expansion_search import find_pair
from scripts.level0_expansion_sources import load_pool
from scripts.prepare_level0_axis_review import board, opaque_id, pair_preview
from scripts.prepare_level0_axis_v2 import DEFAULT_OUT, read_rows, seal
from scripts.prepare_obvious_defect_review import DEFAULT_CONFIG, ROOT, STATES, checker_result, sha256_path, write_jsonl
from turnitover.assets.audit_urdf import audit_model
from turnitover.assets.urdf import UrdfModel
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

DEFAULT_BASE = DEFAULT_OUT
DEFAULT_EXPANSION = ROOT / 'docs/evidence/partnet-mobility-level0-axis-v3'


def run(out=DEFAULT_EXPANSION, base=DEFAULT_BASE, target=60, catalog=None, download_manifest=None):
    out, base = Path(out).resolve(), Path(base).resolve()
    if out == base or base in out.parents or out in base.parents:
        raise ValueError('Expansion output must be separate from the immutable baseline')
    if not base.exists():
        raise ValueError('Provide an existing reviewed baseline bundle')
    if not (out / 'private/capacity_audit.json').is_file():
        raise ValueError('Prepare private/capacity_audit.json before expansion')
    base_manifest_sha = sha256_path(base / 'manifest.json')
    old = json.loads((base / 'manifest.json').read_text())
    for name, digest in old['files'].items():
        if sha256_path(base / name) != digest:
            raise ValueError(f'Base SHA mismatch: {name}')
    original_gold = read_rows(base / 'private/gold.jsonl')
    original_ids = {g['asset_id'] for g in original_gold}
    if not original_ids or len(original_gold) != 2 * len(original_ids):
        raise ValueError('Baseline must contain reviewed clean/axis object pairs')
    for aid in original_ids:
        pair = [g for g in original_gold if g['asset_id'] == aid]
        if {g['kind'] for g in pair} != {'clean', 'joint_axis'} or any(
                g.get('training_eligibility') != 'eligible' for g in pair):
            raise ValueError(f'Baseline pair is not training eligible: {aid}')
    if target < len(original_ids):
        raise ValueError('Expansion target cannot be smaller than the baseline')
    if not out.exists():
        shutil.copytree(base, out)
    elif not (out / 'private/gold.jsonl').exists():
        files = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
        allowed = {'private/capacity_audit.json', 'SOURCE_AUDIT.md'}
        if not files <= allowed:
            raise ValueError('Existing output is neither a bundle nor an audit-only directory')
        # The audit belongs to this expansion, even when the baseline contains
        # an identically named audit and report from its own generation.
        audit_files = {name: (out / name).read_bytes() for name in files}
        shutil.copytree(base, out, dirs_exist_ok=True)
        for name, content in audit_files.items():
            (out / name).write_bytes(content)
    elif read_rows(out / 'private/gold.jsonl')[:len(original_gold)] != original_gold:
        raise ValueError('Output does not preserve the reviewed base')
    cfg = load_generate_config(DEFAULT_CONFIG)
    if (cfg.render.width, cfg.render.height) != (256, 256):
        raise ValueError('Level 0 requires native 256x256 rendering')
    source_options = {}
    if catalog is not None:
        source_options['catalog_path'] = catalog
    if download_manifest is not None:
        source_options['download_path'] = download_manifest
    pool, original_splits, provenance = load_pool(**source_options)
    capacity_path = out / 'private/capacity_audit.json'
    capacity = json.loads(capacity_path.read_text()) if capacity_path.exists() else None
    if capacity is not None:
        for name, digest in capacity['inputs'].items():
            if sha256_path(Path(name)) != digest:
                raise ValueError(f'Capacity audit input SHA mismatch: {name}')
        queued = {f"PM_{row['asset_id']}": row for row in capacity['queue']}
        for candidate in pool:
            row = queued.get(candidate['spec'].asset_id)
            if row is not None:
                meta = candidate['metadata']
                if any(row[key] != meta[field] for key, field in (
                        ('model_id', 'model_id'), ('archive_sha256', 'archive_sha256'),
                        ('spec_sha256', 'original_spec_sha256'),
                        ('geometry_fingerprint', 'original_geometry_fingerprint'))):
                    raise ValueError(f'Capacity queue identity mismatch: {candidate["spec"].asset_id}')
    metadata = {r['spec'].asset_id: r['metadata'] for r in pool}
    # The completed baseline owns its provenance, even if it left the live pool.
    base_catalog = json.loads((base / old['source_catalog']).read_text())
    clean_by_spec = {f"../{g['case_id']}.json": g['asset_id']
                     for g in original_gold if g['kind'] == 'clean'}
    for entry in base_catalog['assets']:
        metadata[clean_by_spec[entry['spec']]] = dict(entry)
    if 'expanded_splits' in old:
        original_splits = json.loads((base / old['expanded_splits']).read_text())
    public = read_rows(out / 'review_manifest.jsonl')
    gold = read_rows(out / 'private/gold.jsonl')
    checks = read_rows(out / 'private/checks.jsonl')
    reviews = read_rows(out / 'human_review.jsonl')
    selected = {g['asset_id'] for g in gold}
    models = {metadata[a]['model_id'] for a in selected}
    fingerprints = {metadata[a]['original_geometry_fingerprint'] for a in selected}
    checkpoint_path = out / 'private/expansion_screening.json'
    screening = json.loads(checkpoint_path.read_text())['assets'] if checkpoint_path.exists() else []
    attempted = {a['asset_id'] for a in screening}
    for name in ('private/screening.json', 'private/inherited_screening.json'):
        path = base / name
        if path.exists():
            attempted.update(a['asset_id'] for a in json.loads(path.read_text())['assets'])
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
            if capacity is not None and spec.asset_id not in queued:
                continue
            if (meta.get('previous_motion_review') or {}).get('decision') == 'failed':
                screening.append({'asset_id': spec.asset_id, 'status': 'rejected',
                    'rejection_reason_counts': {'historical_motion_review_failed': 1},
                    'previous_motion_review': meta['previous_motion_review']})
                continue
            if meta['model_id'] in models or meta['original_geometry_fingerprint'] in fingerprints:
                screening.append({'asset_id': spec.asset_id, 'status': 'duplicate_source_object'})
                continue
            pair, diagnostics = find_pair(session, ObjectProgram.from_spec(spec), views, 1000000 + index * 50000)
            authenticity = None
            if pair:
                source = Path(meta.get('source_urdf', ROOT / 'data/partnet-mobility-pilot/extracted' /
                                       str(meta['source_asset_id']) / 'mobility.urdf'))
                try:
                    authenticity = audit_model(session, UrdfModel(source), spec)
                except Exception as error:
                    # Keep the raw failure; conversion fidelity is never inferred
                    # from the selected camera or the corruption checker.
                    authenticity = {'passed': False, 'source_urdf': str(source),
                        'error_type': type(error).__name__, 'error': str(error)}
                if authenticity.get('passed') is not True:
                    pair = None
            result = {'asset_id': spec.asset_id, 'category': spec.category,
                'original_train': candidate['original_train'],
                'status': 'automatic_pass' if pair else 'rejected', 'joints': diagnostics}
            if authenticity is not None:
                result['source_authenticity'] = authenticity
                if authenticity.get('passed') is not True:
                    result['rejection_reason_counts'] = {'source_authenticity_failed': 1}
            screening.append(result)
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
    # Historical hashes remain in expansion_base_manifest; this new version pins
    # the actual implementation used for its own generation and validation.
    inputs.update({name: sha256_path(Path(name)) for name in inputs
                   if Path(name).suffix == '.py'})
    inputs.update(provenance)
    inputs.update(capacity['inputs'])
    inputs[str(capacity_path.resolve())] = sha256_path(capacity_path)
    audit_script = Path(capacity.get('audit_script', ROOT / 'scripts/audit_level0_expansion_capacity.py'))
    inputs[str(audit_script.resolve())] = sha256_path(audit_script)
    inputs.update({str(p.resolve()): sha256_path(p) for p in
        (Path(__file__), ROOT / 'scripts/level0_expansion_search.py', ROOT / 'scripts/level0_expansion_sources.py',
         ROOT / 'turnitover/assets/audit_urdf.py', ROOT / 'turnitover/assets/urdf.py')})
    seal(out, inputs)
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest.update(source_catalog='private/source_catalog/catalog.json', expanded_splits='private/expanded_splits.json',
        target_objects=target, base_objects=len(original_ids), base_manifest_sha256=base_manifest_sha,
        base_directory=str(base), expansion_protocol='native-endpoint-motion; normal-only frozen single camera; unchanged pixel gates')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    from scripts.finalize_level0_axis_v2 import finalize
    result = finalize(out)
    if sha256_path(base / 'manifest.json') != base_manifest_sha:
        raise ValueError('Immutable baseline manifest changed')
    for name, digest in old['files'].items():
        if sha256_path(base / name) != digest:
            raise ValueError(f'Immutable baseline changed: {name}')
    if read_rows(out / 'private/gold.jsonl')[:len(original_gold)] != original_gold:
        raise ValueError('Inherited Gold changed during finalization')
    for name, digest in old['files'].items():
        if name.startswith(('boards/', 'inputs/', 'frames/')) or (name.startswith('private/l0-') and name.endswith('.json')):
            if sha256_path(out / name) != digest:
                raise ValueError(f'Inherited artifact changed during finalization: {name}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_EXPANSION)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--target', type=int, default=60)
    parser.add_argument('--catalog', type=Path, help='Independent source catalog; defaults to the configured pool')
    parser.add_argument('--download-manifest', type=Path, help='Archive manifest for the selected source pool')
    args = parser.parse_args()
    print(json.dumps(run(args.out.resolve(), args.base.resolve(), args.target,
                         catalog=args.catalog, download_manifest=args.download_manifest)))
