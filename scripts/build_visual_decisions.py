"""Render three non-oracle view orders; export public-image-only v5 decisions. No APIs."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from scripts.run_oracle_supervision import _attach_identity, _benchmark_provenance
from turnitover.assets.source import make_source
from turnitover.config import load_generate_config
from turnitover.core.actions import RequestView, Detent, detent_value
from turnitover.core.program import ObjectProgram
from turnitover.dataset.prepare import records
from turnitover.oracle.evidence import replay_corruptions
from turnitover.oracle.supervision import action_json
from turnitover.oracle.motion_decisions import schedules, states, advance
from turnitover.oracle.visual_decisions import VIEWS, ORDERS, image_evidence, make_row, deduplicate
from turnitover.policy.loop import execute_observation
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    inputs = {}
    provenance = _benchmark_provenance(args.benchmark)
    for split in ('train', 'validation'):
        for line in (args.benchmark / 'prepared' / f'{split}.inputs.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['id'] in inputs:
                raise ValueError('Duplicate split identity')
            inputs[row['id']] = split, row['context']
    cfg = load_generate_config(args.config, repo_root=args.asset_workspace)
    cfg = replace(cfg, render=replace(cfg.render, harness_dir=args.harness_workspace / 'web/dist',
                                     esbuild_bin=args.harness_workspace / 'web/node_modules/.bin/esbuild'))
    source = make_source(**cfg.asset_source)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'version': 5, 'status': 'incomplete', 'formal_training': True,
                'scope': 'articulated_shape_consistency_with_public_matched_state_references' if args.motion else 'static_shape_consistency_with_six_public_rest_references',
                'budget': 6, 'orders': 'fixed_motion_schedules' if args.motion else ORDERS, 'collection_policy': 'fixed_order_full_budget_off_policy',
                'source_hashes': {str(p.resolve()): sha(p) for p in [*provenance, args.config, Path(__file__),
                                  Path('turnitover/oracle/visual_decisions.py'), Path('turnitover/oracle/motion_decisions.py')]},
                'label_inputs': 'public aligned reference and current prefix PNGs only',
                'limitations': ['No defect-type, pass, natural-error or optimal action supervision; motion uses only the first public joint.',
                                'Silhouette gate is deterministic visual evidence, not independent human diagnosis.',
                                'Collection continues after sufficient evidence to sample late decision states.']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    streams = {s: (out / f'{s}.stopping.jsonl').open('w') for s in ('train', 'validation')}
    n = 0
    try:
        with (out / 'trajectories.jsonl').open('w') as traces, ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
            manifest['browser_version'] = session.browser_version()
            for sample, _ in records(sorted((args.benchmark / 'raw').glob('*.tar'))):
                uid = hashlib.sha256(f'{sample.source_id}:{sample.sample_id}'.encode()).hexdigest()[:24]
                if uid not in inputs:
                    continue
                split, context = inputs[uid]
                reference = ObjectProgram.from_spec(source.get(sample.asset_id))
                # Gold is used only to reconstruct the already frozen candidate,
                # never passed to evidence/decision/order functions.
                candidate = replay_corruptions(reference, sample.corruptions)
                if (reference.sha, candidate.sha) != (sample.reference_sha, sample.program_sha):
                    raise ValueError('Source replay mismatch')
                folder = out / 'images' / uid
                folder.mkdir(parents=True)
                if args.motion:
                    joint = reference.spec.joints[0]
                    detent = 'limit' if joint.limits[1] != 0 else 'zero'
                    reference_states = states(joint.id, detent)
                    orders = schedules(joint.id, detent)
                    context = {**context, 'reference_states': reference_states}
                else:
                    reference_states = [{'view': v} for v in VIEWS]
                    orders = [tuple(RequestView(v) for v in order) for order in ORDERS]
                refs, ref_pixels, ref_cameras = [], {}, {}
                session.load(reference, framing=sample.framing)
                for ref_index, state in enumerate(reference_states):
                    if args.motion:
                        session.reset_joints()
                        session.set_joints({joint.id: detent_value(Detent(state['detent']), joint.limits)})
                    obs = session.request_view(state['view'])
                    path = f'images/{uid}/reference-{ref_index}.png'
                    (out / path).write_bytes(obs.png)
                    refs.append(path)
                    ref_pixels[ref_index], ref_cameras[ref_index] = obs.png, obs.camera
                session.dispose()
                identity_record = {'images': refs}
                _attach_identity(identity_record, out, sample, source.metadata(sample.asset_id), reference, candidate)
                for order_id, order in enumerate(orders):
                    history, evidence = [], []
                    session.load(candidate, framing=sample.framing)
                    state = reference_states[0]
                    for step, action in enumerate(order):
                        state = advance(state, action_json(action)) if args.motion else {'view': action.view_id}
                        ref_index = reference_states.index(state)
                        captured = []
                        def sink(index, png):
                            path = f'images/{uid}/order{order_id}-{index}.png'
                            (out / path).write_bytes(png)
                            captured.append(png)
                            return path
                        obs = execute_observation(session, action, step, sink)
                        payload = {k: v for k, v in obs.payload.items() if k != 'render_ms'}
                        if 'camera' in payload and payload['camera'] != ref_cameras[ref_index]:
                            raise ValueError('Unaligned camera')
                        history.append({'step': step, 'action': action_json(action),
                                        'image_ref': obs.image_ref, 'payload': payload})
                        if args.motion:
                            history[-1]['reference_index'] = ref_index
                        evidence.append(image_evidence(ref_pixels[ref_index], captured[0]))
                        row = make_row(uid, order_id, context, refs, history[:], evidence[:], identity_record['data_identity'])
                        if row:
                            row['image_sha256'] = [sha(out / p) for p in row['images']]
                            streams[split].write(json.dumps(row) + '\n')
                    session.dispose()
                    traces.write(json.dumps({'id': f'{uid}:order{order_id}', 'split': split,
                                 'data_identity': identity_record['data_identity'], 'references': refs,
                                 'history': history, 'evidence': evidence, 'context': context}) + '\n')
                n += 1
                print(json.dumps({'rendered_samples': n, 'asset': sample.asset_id}), flush=True)
        manifest.update(status='complete', samples=n)
    finally:
        for f in streams.values():
            f.close()
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    reexport(out)


def reexport(root):
    """Rebuild decisions from completed real traces without any new observation."""
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Complete traces required')
    rows = {'train': [], 'validation': []}
    for line in (root / 'trajectories.jsonl').read_text().splitlines():
        t = json.loads(line)
        uid, order = t['id'].split(':order')
        evidence = []
        for i, h in enumerate(t['history']):
            ref_index = h.get('reference_index', VIEWS.index(h['action']['view_id']) if h['action']['type'] == 'request_view' else None)
            evidence.append(image_evidence((root / t['references'][ref_index]).read_bytes(),
                                           (root / h['image_ref']).read_bytes()))
            row = make_row(uid, int(order), t['context'], t['references'], t['history'][:i+1],
                           evidence[:], t['data_identity'])
            if row:
                row['image_sha256'] = [sha(root / p) for p in row['images']]
                rows[t['split']].append(row)
    manifest['duplicates_removed'] = {s: len(rs)-len(deduplicate(rs)) for s, rs in rows.items()}
    for split, records_ in rows.items():
        records_ = deduplicate(records_)
        (root / f'{split}.stopping.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records_))
    manifest['decision_source_sha256'] = sha(Path('turnitover/oracle/visual_decisions.py'))
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reexport', type=Path)
    p.add_argument('--motion', action='store_true')
    p.add_argument('--benchmark', type=Path)
    p.add_argument('--asset-workspace', type=Path)
    p.add_argument('--harness-workspace', type=Path)
    p.add_argument('--config', type=Path, default=Path('configs/generate_partnet_mobility_supervision.yaml'))
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.reexport:
        reexport(args.reexport)
    elif all((args.benchmark, args.asset_workspace, args.harness_workspace, args.out)):
        run(args)
    else:
        p.error('Rendering requires benchmark, asset-workspace, harness-workspace and out')
