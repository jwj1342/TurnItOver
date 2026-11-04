"""Build isolated pending-human Medium pairs with unchanged strength/tightening gates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.medium_expansion_sources import load_sources, sha
from scripts.tier_construction.medium_strength_artifacts import artifacts, object_id, page
from scripts.tier_construction.medium_strength_search import find_pair, POLICY as GENERATION_POLICY
from scripts.tier_construction.medium_tightening_policy import evaluate, POLICY as ADMISSION_POLICY
from scripts.tier_construction.prepare_medium_pilot import _json, _jsonl
from scripts.tier_construction.prepare_obvious_defect_review import sha256_path
from turnitover.assets.audit_urdf import audit_model
from turnitover.assets.urdf import UrdfModel
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views


def checkpoint(out, records, pins, target):
    accepted = [row for row in records if row['status'] == 'automatic_pass']
    for name, key in (('review_manifest.jsonl', 'public'), ('private/gold.jsonl', 'gold'),
                      ('private/checks.jsonl', 'checks'), ('human_review.jsonl', 'reviews')):
        _jsonl(out / name, sorted((case for row in accepted for case in row[key]), key=lambda r: r['case_id']))
    public = [case for row in accepted for case in row['public']]
    public.sort(key=lambda row: hashlib.sha256(('medium-review-v1:' + row['case_id']).encode()).hexdigest())
    page(out, public, 'review.html')
    report = dict(status='automatic_candidates_ready' if len(accepted) >= target else 'collection_incomplete',
        target_new_objects=target, independent_objects=len(accepted), cases=len(public),
        attempted_sources=len(records), human_review_status='pending_human_review',
        training_eligible_objects=0, configs=dict(Counter(row['config'] for row in accepted)),
        policies=dict(generation=GENERATION_POLICY, admission=ADMISSION_POLICY))
    _json(out / 'validation.json', report)
    _json(out / 'manifest.json', dict(version=1, inputs=pins,
        files={p.relative_to(out).as_posix(): sha256_path(p) for p in sorted(out.rglob('*'))
               if p.is_file() and p.name != 'manifest.json'}))
    return report


def build(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pool, pins = load_sources(args.source, args.exclude, args.splits)
    for path in [Path(__file__), Path(__file__).with_name('medium_expansion_sources.py'), args.config,
                 *Path(__file__).with_name('tier_construction').glob('*.py')]:
        pins[str(path.resolve())] = sha(path)
    audit = dict(version=1, source_identities=[row['metadata']['source_identity'] for row in pool],
        asset_ids=[row['spec'].asset_id for row in pool], inputs=pins,
        human_review_status='pending_human_review', selection='new multi-joint objects only')
    audit_path = out / 'private/source_audit.json'
    if audit_path.exists() and json.loads(audit_path.read_text()) != audit:
        raise ValueError('Source inputs changed since frozen queue creation; use a new output directory')
    _json(audit_path, audit)
    if args.prepare_only:
        return dict(source_candidates=len(pool), source_audit=str(audit_path))
    records = [json.loads(p.read_text()) for p in sorted((out / 'private/objects').glob('*/record.json'))]
    records += [json.loads(p.read_text()) for p in sorted((out / 'private/rejections').glob('*.json'))]
    done = {row['asset_id'] for row in records}
    cfg = load_generate_config(args.config)
    if (cfg.render.width, cfg.render.height) != (256, 256):
        raise ValueError('Requires native 256x256')
    views = load_views(cfg.views_path)
    pins[str(cfg.views_path.resolve())] = sha(cfg.views_path)
    with ObservationSession(cfg.render, views) as session:
        for item in pool:
            if sum(row['status'] == 'automatic_pass' for row in records) >= args.target or len(done) >= args.max_tries:
                break
            spec = item['spec']
            if spec.asset_id in done:
                continue
            reference = ObjectProgram.from_spec(spec)
            all_diagnostics, pair, admission, authenticity = {}, None, None, None
            configuration = None
            for configuration in ('multi_joint_axis', 'multi_direction_flip'):
                seed = 17_000_000 + int(object_id(configuration, spec.asset_id)[:8], 16)
                pair, diagnostics = find_pair(session, reference, views, configuration, seed)
                all_diagnostics[configuration] = diagnostics
                if not pair:
                    continue
                ref, chosen, joint_id, values = pair
                label = chosen['label']
                gold = dict(kind=label.defect_id.rsplit('.', 1)[1], joint_id=joint_id,
                            parts=list(label.parts), parameters=label.params, state_values=values)
                frames = [{state: data['frames'][state][chosen['view']] for state in values}
                          for data in (ref, chosen['render'])]
                admission = evaluate(to_dict(reference.spec), to_dict(chosen['program'].spec), gold, *frames)
                if admission['reasons']:
                    pair = None
                    continue
                break
            if pair:
                authenticity = audit_model(session, UrdfModel(Path(item['source_urdf'])), spec)
                if authenticity.get('passed') is not True:
                    pair = None
            record = dict(asset_id=spec.asset_id, category=spec.category, config=configuration,
                status='automatic_pass' if pair else 'rejected', diagnostics=all_diagnostics,
                source_authenticity=authenticity, source=item['metadata'], files={},
                tightening_admission=admission, collection_phase='medium-tier-expansion-v1')
            if pair:
                item['config'] = configuration
                record.update(artifacts(out, item, pair, reference))
                _, chosen, _, values = pair
                record['camera'] = dict(view=chosen['view'], framing=to_dict(chosen['framing']),
                    state_values=values, policy='normal-only frozen before corruption; no error-guided view retry')
                record['metrics'] = chosen['metrics']
                case_ids = {row['case_id'] for row in record['public']}
                record['files'] = {p.relative_to(out).as_posix(): sha256_path(p) for p in out.rglob('*')
                    if p.is_file() and any(cid in p.as_posix() for cid in case_ids)}
            path = (out / 'private/objects' / spec.asset_id / 'record.json' if pair else
                    out / 'private/rejections' / (spec.asset_id + '.json'))
            _json(path, record)
            records.append(record)
            done.add(spec.asset_id)
            report = checkpoint(out, records, pins, args.target)
            print(json.dumps(dict(asset_id=spec.asset_id, status=record['status'],
                                  objects=report['independent_objects'], tried=len(done))), flush=True)
    for name, digest in pins.items():
        if sha(name) != digest:
            raise ValueError('Input changed during construction: ' + name)
    return checkpoint(out, records, pins, args.target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--source', type=Path, nargs=2, action='append', required=True, metavar=('CATALOG', 'DOWNLOAD'))
    parser.add_argument('--exclude', type=Path, action='append', required=True)
    parser.add_argument('--target', type=int, default=75)
    parser.add_argument('--max-tries', type=int, default=1000)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args)), flush=True)


if __name__ == '__main__':
    main()
