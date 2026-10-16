"""One pilot then seventeen single-Episode active Kimi runs; no retry or training."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.easy_teacher_audit import audit_case, validate_export
from scripts.easy_teacher_data import BUNDLE, COHORT, PILOT, PROTOCOL, prepare, read_rows
from scripts.prepare_obvious_defect_review import DEFAULT_CONFIG, ROOT, sha256_path, write_jsonl
from turnitover.assets.catalog import decode_spec
from turnitover.config import load_generate_config
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.core.sample import Framing
from turnitover.models.config import model_config, read_environment
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.teacher.episode import run_teacher_episode
from turnitover.teacher.export import export_episodes
from turnitover.teacher.policy import TeacherPolicy
from turnitover.teacher.providers import KimiTeacher
from turnitover.verifier.contracts import Context

DEFAULT_OUT = ROOT / 'docs/evidence/kimi-easy-interactive-v2'
BUDGET = 8  # one common hierarchy observation plus seven model-selected observations


def configure(out):
    preflight = prepare(out)
    inputs = [COHORT, BUNDLE / 'manifest.json', BUNDLE / 'human_review.jsonl',
              Path(__file__), ROOT / 'scripts/easy_teacher_data.py', ROOT / 'scripts/easy_teacher_audit.py',
              ROOT / 'turnitover/teacher/prompt.txt', ROOT / 'turnitover/teacher/part_convention.txt']
    config = {'experiment': 'kimi-easy-interactive-v2', 'model': 'k3-256k',
              'budget_including_common_hierarchy': BUDGET, 'max_api_calls_per_case': BUDGET,
              'global_request_cap': 18 * BUDGET, 'decision_attempts': 1,
              'pilot_case': PILOT, 'episodes_per_case': 1, 'prompt_tuning': False,
              'scope': 'one normal-selected joint and fixed human-reviewed view per object',
              'reference_protocol': PROTOCOL, 'preflight': preflight,
              'transmission_review_basis': 'user-authorized-18-case-PartNet-Train-interactive-teacher-experiment',
              'input_sha256': {str(p.relative_to(ROOT)): sha256_path(p) for p in inputs},
              'boundaries': {'test_read_or_modified': False, 'train_started': False, 'old_117_merged': False}}
    write_json_atomic(out / 'config.json', config)
    return config


def collect(out, phase, env_file):
    config = json.loads((out / 'config.json').read_text())
    for relative, digest in config['input_sha256'].items():
        if sha256_path(ROOT / relative) != digest:
            raise ValueError(f'Frozen experiment input changed: {relative}')
    prepared = read_rows(out / 'prepared.jsonl')
    if phase == 'pilot':
        selected = [r for r in prepared if r['case_id'] == PILOT]
    else:
        pilot = json.loads((out / 'cases' / PILOT / 'audit.json').read_text())
        if not pilot['pipeline_ok']:
            raise ValueError('Pilot failed pipeline validation; collection stopped')
        selected = [r for r in prepared if r['case_id'] != PILOT]
    gold = {r['id']: r for r in read_rows(out / 'private/gold.jsonl')}
    annotations = {r['id']: r for r in read_rows(out / 'private/annotations.jsonl')}
    model = model_config('teacher', read_environment(env_file))
    model.validate()
    if model.model != 'k3-256k':
        raise ValueError('This experiment requires exactly k3-256k')
    cfg = load_generate_config(DEFAULT_CONFIG)
    views = load_views(cfg.views_path)
    with ObservationSession(cfg.render, views) as session:
        for item in selected:
            cid = item['case_id']
            folder = out / 'cases' / cid
            if folder.exists():
                raise ValueError(f'Refusing to retry an existing paid Episode: {cid}')
            folder.mkdir(parents=True)
            path = out / item['manifest']
            manifest = json.loads(path.read_text())
            refs = []
            for state in manifest['reference_protocol']['states']:
                target = folder / state['file']
                shutil.copyfile(path.parent / state['file'], target)
                assert sha256_path(target) == state['sha256']
                refs.append(target)
            spec = decode_spec(json.loads((BUNDLE / f'private/{cid}.json').read_text()))
            framing = manifest['framing']
            session.load(ObjectProgram.from_spec(spec), Framing(tuple(framing['center']), framing['radius']))
            session.request_view(item['context']['views'][0])  # initialize declared camera; no image supplied
            teacher = KimiTeacher(model, None, transmission_reviewed=True,
                                  estimated_input_tokens=20000, max_real_requests=BUDGET)
            policy = TeacherPolicy(teacher, folder, refs,
                                   reference_states=[r['label'] for r in manifest['reference_protocol']['states']],
                                   protocol_addenda=('part_convention.txt',))
            identity = {**item['identity'], 'third_party_transmission_reviewed': True,
                        'transmission_review_basis': config['transmission_review_basis']}
            # identity fields verified by the auditor are a subset; no private label reaches policy.
            episode = run_teacher_episode(session, policy, Context(**item['context']), BUDGET, folder,
                episode_id=cid, max_model_calls=BUDGET, decision_attempts=1,
                data_identity=identity, initial_actions=({'type': 'query_runtime', 'property': 'hierarchy'},))
            episode.update(reference_protocol=PROTOCOL,
                           reference_states=[r['label'] for r in manifest['reference_protocol']['states']],
                           runtime_joint_limits=manifest['candidate_joint_limits'],
                           frozen_witness_manifest={'path': str(path.resolve()), 'sha256': sha256_path(path)},
                           real_api_requests=teacher.calls, usage_cost_rmb_estimate=None, billing_cost_rmb=None)
            write_json_atomic(folder / 'episode.json', episode)
            result = audit_case(folder, item, gold[cid], annotations[cid], manifest)
            session.dispose()
            print(json.dumps({'phase': phase, **result}), flush=True)


def finalize(out):
    items = read_rows(out / 'prepared.jsonl')
    sources, results = [], []
    for item in items:
        folder = out / 'cases' / item['case_id']
        results.append(json.loads((folder / 'result.json').read_text()))
        sources.append(folder)
    export = export_episodes(sources, out / 'sft', audit_name='export_audit.json')
    loader = validate_export(out / 'sft')
    write_jsonl(out / 'case_results.jsonl', results)
    write_jsonl(out / 'uncertain.jsonl', [r for r in results if r['verdict'] == 'uncertain'])
    summary = {'cases': len(results), 'complete_episodes': sum(r['complete_episode'] for r in results),
               'strict_correct': sum(r['strict_correct'] for r in results),
               'by_kind': {k: {'cases': sum(r['kind'] == k for r in results),
                              'complete': sum(r['kind'] == k and r['complete_episode'] for r in results),
                              'strict_correct': sum(r['kind'] == k and r['strict_correct'] for r in results)}
                           for k in ('clean', 'joint_axis', 'part_offset')},
               'api_calls': sum(r['api_calls'] for r in results),
               'token_usage': {k: sum(r['token_usage'][k] for r in results)
                               for k in ('input_tokens', 'output_tokens', 'total_tokens')},
               'usage_incomplete_calls': sum(r['usage_incomplete_calls'] for r in results),
               'billing_cost_rmb': None, 'export': export['counts'], 'loader': loader,
               'training_started': False, 'old_117_merged': False, 'test_modified': False}
    write_json_atomic(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--phase', choices=('prepare', 'pilot', 'remaining', 'finalize'), required=True)
    args = parser.parse_args()
    if args.phase == 'prepare':
        print(json.dumps(configure(args.out)))
    elif args.phase == 'finalize':
        finalize(args.out)
    else:
        collect(args.out, args.phase, args.env_file)
