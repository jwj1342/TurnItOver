"""Add strict pending-only Easy defects to explicitly selected baseline Clean-only sources."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.tier_construction.construction_pilot import find_case, write_frames
from scripts.tier_construction.prepare_level0_axis_review import board, pair_preview
from scripts.tier_construction.prepare_obvious_defect_review import checker_result
from turnitover.assets.audit_urdf import audit_model
from turnitover.assets.catalog import decode_spec
from turnitover.assets.urdf import UrdfModel
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

IDS = ('PM_10269', 'PM_9992', 'PM_10357', 'PM_11581', 'PM_10305')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    if args.out.exists():
        raise ValueError('Output must be new; historical Clean cases are never overwritten')
    args.out.mkdir(parents=True)
    with zipfile.ZipFile(args.baseline) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/private/gold.jsonl'))
        gold = [json.loads(line) for line in archive.read(name).splitlines()]
    inputs = {str(args.baseline.resolve()): sha(args.baseline), str(args.config.resolve()): sha(args.config),
              str(Path(__file__).resolve()): sha(__file__)}
    for path in Path(__file__).with_name('tier_construction').glob('*.py'):
        inputs[str(path.resolve())] = sha(path)
    sources = {}
    for catalog_path, download_path in args.source:
        catalog_path, download_path = catalog_path.resolve(), download_path.resolve()
        conversion_path = catalog_path.parent / 'conversion.json'
        for path in (catalog_path, conversion_path, download_path):
            inputs[str(path)] = sha(path)
        conversion = json.loads(conversion_path.read_text())
        for row in conversion['assets']:
            aid = 'PM_' + str(row['asset_id'])
            if aid not in IDS or row['status'] != 'structural_candidate':
                continue
            clean = [g for g in gold if g['asset_id'] == aid]
            if len(clean) != 1 or clean[0]['kind'] != 'clean':
                raise ValueError('Expected exactly one historical Clean case: ' + aid)
            path = catalog_path.parent / (aid + '.json')
            if sha(path) != clean[0]['original_spec_sha256'] or row['archive_sha256'] != clean[0]['source_archive_sha256']:
                raise ValueError('Source does not match baseline Clean hashes: ' + aid)
            download = json.loads(download_path.read_text())
            archive_row = next(a for a in download['archives'] if str(a['asset_id']) == str(row['asset_id']))
            source_archive = download_path.parent / archive_row['path']
            if sha(source_archive) != row['archive_sha256']:
                raise ValueError('Official source archive SHA mismatch')
            urdf = download_path.parent / 'extracted' / str(row['asset_id']) / 'mobility.urdf'
            for path in (path, source_archive, urdf):
                inputs[str(path)] = sha(path)
            sources[aid] = dict(row=row, spec=catalog_path.parent / (aid + '.json'), urdf=urdf, clean=clean[0])
    if set(sources) != set(IDS):
        raise ValueError('Missing baseline source objects')
    dump(args.out / 'private/construction-inputs.json', inputs)
    cfg = load_generate_config(args.config)
    views = load_views(cfg.views_path)
    public, new_gold, checks, records = [], [], [], []
    with ObservationSession(cfg.render, views) as session:
        for aid in IDS:
            item = sources[aid]
            reference = ObjectProgram.from_spec(decode_spec(json.loads(item['spec'].read_text())))
            if reference.sha != item['clean']['reference_sha']:
                raise ValueError('Baseline reference ABI hash differs: ' + aid)
            diagnostics, pair = {}, None
            # Deterministic independent seed batches retain every strict gate and normal-only camera.
            for batch in range(args.seed_batches):
                for defect in ('kinematics.joint_axis', 'kinematics.direction_flip'):
                    pair, detail = find_case(session, reference, views, {'defect': defect},
                                             41_000_000 + IDS.index(aid) * 1_000_000 + batch * 100_000)
                    diagnostics[f'{batch}:{defect}'] = detail
                    if pair:
                        break
                if pair:
                    break
            authenticity = audit_model(session, UrdfModel(item['urdf']), reference.spec) if pair else None
            if authenticity and authenticity.get('passed') is not True:
                pair = None
            record = dict(asset_id=aid, source_identity=item['row']['source_identity'],
                historical_clean_case_id=item['clean']['case_id'], source_authenticity=authenticity,
                status='automatic_pass' if pair else 'rejected', diagnostics=diagnostics)
            if pair:
                ref, chosen, joint_id, values = pair
                cid = 'ec-' + hashlib.sha256((aid + ':' + str(chosen['seed'])).encode()).hexdigest()[:16]
                rendered, view, label = chosen['render'], chosen['view'], chosen['label']
                result = checker_result(chosen['program'], rendered, reference, ref)
                if result['code_correct'] or not result['checks'][1]['passed']:
                    raise ValueError('Independent export checker disagreement')
                board(args.out / f'boards/{cid}.png', cid, ref, rendered, view)
                pair_preview(args.out / f'inputs/{cid}.png', ref, rendered, view)
                write_frames(args.out, cid, ref, rendered, view)
                for role, frames in (('reference', ref), ('candidate', rendered)):
                    for state in ('rest', 'mid', 'limit'):
                        path = args.out / f'native/{cid}/{role}_{state}.png'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(frames['frames'][state][view])
                program_path = f'private/programs/{cid}.json'
                dump(args.out / program_path, to_dict(chosen['program'].spec))
                public.append(dict(case_id=cid, status='pending_human_review', board=f'boards/{cid}.png',
                    images=[f'frames/{cid}/{i}.png' for i in range(4)], image_size=[256,256]))
                new_gold.append(dict(case_id=cid, asset_id=aid, source_identity=item['row']['source_identity'],
                    source_archive_sha256=item['row']['archive_sha256'], original_spec_sha256=item['row']['spec_sha256'],
                    original_geometry_fingerprint=item['row']['scale_audit']['geometry_fingerprint'],
                    historical_clean_case_id=item['clean']['case_id'], kind=label.defect_id.rsplit('.',1)[1],
                    defect_id=label.defect_id, joint_id=joint_id, parts=list(label.parts), parameters=label.params,
                    seed=chosen['seed'], candidate_sha=chosen['program'].sha, reference_sha=reference.sha,
                    state_values=values, view=view, framing=to_dict(chosen['framing']), program=program_path,
                    training_eligibility='pending_human_review'))
                checks.append(dict(case_id=cid, **result))
                record['case_id'] = cid
            records.append(record)
            dump(args.out / f'private/records/{aid}.json', record)
            for name, rows in (('review_manifest.jsonl',public),('private/gold.jsonl',new_gold),('private/checks.jsonl',checks)):
                path=args.out/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
            dump(args.out/'validation.json',dict(requested_objects=5,attempted=len(records),
                automatic_pass=sum(r['status']=='automatic_pass' for r in records),human_review_status='pending_human_review'))
            print(json.dumps(dict(asset_id=aid,status=record['status'])),flush=True)
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Pinned input changed: '+path)
    dump(args.out/'manifest.json',dict(inputs=inputs,files={p.relative_to(args.out).as_posix():sha(p)
        for p in args.out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--source',type=Path,nargs=2,action='append',required=True)
    parser.add_argument('--seed-batches',type=int,default=4)
    run(parser.parse_args())
