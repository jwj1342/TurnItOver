#!/usr/bin/env python3
"""Run one shared API initial, paired native loops, then independent scoring."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from turnitover.api_pipeline.api import FrozenAPI
from turnitover.api_pipeline.fixture import TASK, controlled_cases, generation_prompt, reference_program
from turnitover.api_pipeline.paired import prepare, run_pairs
from turnitover.core.program import ObjectProgram
from turnitover.models.config import model_config, read_environment
from turnitover.render.session import RenderConfig
from turnitover.render.views import load_views

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('single', 'prepare', 'run', 'score', 'recover-final'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--reference-image', type=Path, action='append')
    parser.add_argument('--task-file', type=Path)
    parser.add_argument('--generation-prompt', type=Path)
    parser.add_argument('--initial-program', type=Path)
    parser.add_argument('--policy', choices=('active', 'fixed'), default='active')
    parser.add_argument('--stream', action='store_true',
                        help='SSE transport for compatible APIs; sampling configuration stays unchanged')
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--role', default='teacher', choices=('teacher', 'generator', 'judge'))
    parser.add_argument('--cases', nargs='+', default=['c01', 'c00', 'c02'])
    parser.add_argument('--budget', type=int, default=4)
    parser.add_argument('--revisions', type=int, default=1)
    parser.add_argument('--max-calls', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1,
                        help='Frozen sampling parameter; k3-256k requires exactly 1')
    parser.add_argument('--max-output-tokens', type=int,
                        help='Freeze the same output cap for every comparison and repair call')
    parser.add_argument('--request-timeout', type=float,
                        help='HTTP waiting limit only; identical for both arms')
    args = parser.parse_args()
    render = RenderConfig(ROOT / 'web/dist', ROOT / 'web/node_modules/.bin/esbuild', width=256, height=256)
    views = load_views(ROOT / 'configs/views.yaml')
    if args.stage == 'score':
        from turnitover.api_pipeline.scoring import score_pairs
        if not args.prepared:
            parser.error('--prepared required')
        result = score_pairs(args.prepared, args.out, render, views)
        print(json.dumps({'labels_valid': result['labels_valid'], 'pairs': list(result['pairs'])}))
        return
    config = dataclasses.replace(model_config(args.role, read_environment(args.env_file)), role='shared')
    if args.max_output_tokens is not None:
        if args.max_output_tokens <= 0:
            parser.error('--max-output-tokens must be positive')
        config = dataclasses.replace(config, max_tokens=args.max_output_tokens)
    if args.request_timeout is not None:
        if args.request_timeout <= 0:
            parser.error('--request-timeout must be positive')
        config = dataclasses.replace(config, timeout=args.request_timeout)
    config.validate()
    expected = None
    if args.stage in {'run', 'recover-final'}:
        if not args.prepared:
            parser.error('--prepared required')
        prepared_manifest = json.loads((args.prepared / 'manifest.json').read_text())
        expected = prepared_manifest['returned_model']
    api = FrozenAPI(config, args.out.with_name(args.out.name + '-api'), max_calls=args.max_calls,
                    temperature=args.temperature, expected_model=expected, stream=args.stream)
    if args.stage == 'single':
        if not args.reference_image or not args.task_file:
            parser.error('single requires --reference-image and --task-file')
        from turnitover.repair.iterative import IterativeConfig, run_iterative
        from turnitover.verifier.runner import VerifyConfig
        task = args.task_file.read_text()
        prompt = (args.generation_prompt.read_text() if args.generation_prompt else
                  generation_prompt((ROOT / 'web/src/program_abi.ts').read_text(), task))
        cfg = IterativeConfig(max_runtime_repairs=1, max_visual_revisions=args.revisions, task=task,
            verifier=VerifyConfig(budget=args.budget, mode=args.policy, exact_budget=True))
        result = run_iterative(tuple(args.reference_image), args.out, render, views, cfg,
            api.scoped('single/generator'), generation_prompt=prompt,
            initial_program=ObjectProgram(args.initial_program.read_text()) if args.initial_program else None,
            judge_call=api.scoped('single/judge'), judge_model=config)
        print(json.dumps({'status': result['status'], 'accepted': result['accepted']}))
        return 0 if result['accepted'] else 1
    elif args.stage == 'prepare':
        program = prepare(args.out, render, views, api, TASK,
            generation_prompt((ROOT / 'web/src/program_abi.ts').read_text()), reference_program())
        print(json.dumps({'program_sha': program.sha, 'prepared': str(args.out)}))
    elif args.stage == 'recover-final':
        if not args.source_run:
            parser.error('--source-run required')
        from turnitover.api_pipeline.recovery import recover_final
        result = recover_final(args.prepared, args.source_run, args.out, render, views, api)
        print(json.dumps({'status': result['status'], 'continuation': str(args.source_run)}))
    else:
        base = ObjectProgram((args.prepared / 'shared/final.ts').read_text())
        all_cases = controlled_cases(base)
        if len(set(args.cases)) != len(args.cases) or any(c not in all_cases for c in args.cases):
            parser.error('cases must be unique c00/c01/c02 IDs')
        result = run_pairs(args.prepared, args.out, {c: all_cases[c] for c in args.cases}, render, views, api,
                           budget=args.budget, revisions=args.revisions)
        print(json.dumps({'status': result['status'], 'pairs': list(result['pairs'])}))


if __name__ == '__main__':
    raise SystemExit(main())
