"""Thin paired orchestration over the existing native synthesis loop."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.render.gate import run_render_gate
from turnitover.render.session import ObservationSession
from turnitover.repair.iterative import IterativeConfig, run_iterative
from turnitover.telemetry import git_state, now_iso
from turnitover.verifier.runner import VerifyConfig


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, render, views, api, task, prompt, reference, *, runtime_repairs=1):
    """Reference PNG is newly rendered in Chromium, then a single API initial is shared."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    private = output / 'private'
    private.mkdir()
    (private / 'reference.ts').write_text(reference.source)
    with ObservationSession(render, views) as session:
        session.load(reference, framing=None)
        image = session.request_view(next(v.id for v in views if v.id == 'oblique_fr'))
        (output / 'reference.png').write_bytes(image.png)
        environment = {'browser': session.browser_version(), 'gl': session.gl_info()}
    (output / 'task.txt').write_text(task)
    cfg = IterativeConfig(max_runtime_repairs=runtime_repairs, max_visual_revisions=0,
                          verifier=VerifyConfig(mode='runtime', budget=0), task=task)
    result = run_iterative((output / 'reference.png',), output / 'shared', render, views, cfg,
                           api.scoped('shared/generator'), generation_prompt=prompt)
    if result['status'] != 'verification_uncertain' or not (output / 'shared/final.ts').exists():
        raise ValueError('Shared initial generation/render gate failed; inspect shared/result.json')
    program = ObjectProgram((output / 'shared/final.ts').read_text())
    manifest = {'kind': 'paired_api_prepared', 'created_at': now_iso(), 'model': api.public_config, 'returned_model': api.returned_model,
                'environment': environment, 'git': git_state(Path(__file__).resolve().parents[2]),
                'program_sha': program.sha, 'files': {name: file_sha(output / name) for name in
                    ('reference.png', 'task.txt', 'shared/final.ts', 'private/reference.ts')},
                'render': {'width': render.width, 'height': render.height,
                           'views': [v.to_json() for v in views]}}
    write_json_atomic(output / 'manifest.json', manifest)
    return program


def run_pairs(prepared, output, cases, render, views, api, *, budget=4, revisions=1, runtime_repairs=1):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    manifest = json.loads((prepared / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if file_sha(prepared / name) != expected:
            raise ValueError(f'Prepared input changed: {name}')
    # Initial generation is shared. A comparison freezes one cap for BOTH arms and repairs;
    # its cap may differ from that of the already-completed initial generation.
    frozen = lambda cfg: {k: v for k, v in cfg.items() if k not in {'timeout', 'max_tokens'}}
    if frozen(manifest['model']) != frozen(api.public_config):
        raise ValueError('Prepared and paired API configurations differ')
    render_public = {'width': render.width, 'height': render.height, 'views': [v.to_json() for v in views]}
    if manifest['render'] != render_public:
        raise ValueError('Prepared and paired render configuration differs')
    if not cases or any(key not in {'c00', 'c01', 'c02'} for key in cases):
        raise ValueError('Select nonempty unique procedural case IDs')
    output.mkdir(parents=True, exist_ok=False)
    task = (prepared / 'task.txt').read_text()
    reference = prepared / 'reference.png'
    report = {'kind': 'paired_api_pipeline', 'status': 'running', 'created_at': now_iso(),
              'model': api.public_config, 'returned_model': api.returned_model,
              'prepared_manifest_sha': file_sha(prepared / 'manifest.json'),
              'shared_generation_model': manifest['model'],
              'observation_budget': budget, 'visual_revisions': revisions,
              'runtime_files': {name: file_sha(render.harness_dir / name) for name in ('harness.js', 'three.esm.js')},
              'initials': {}, 'pairs': {}, 'git': git_state(Path(__file__).resolve().parents[2])}
    write_json_atomic(output / 'result.json', report)
    try:
        for index, (case_id, program) in enumerate(cases.items()):
            case_dir = output / case_id
            case_dir.mkdir()
            (case_dir / 'initial.ts').write_text(program.source)
            report['initials'][case_id] = program.sha
            # Check deterministic initial rendering BEFORE either verifier makes paid calls.
            preflight = [case_dir / f'initial-state-{i}' for i in range(2)]
            for folder in preflight:
                gate = run_render_gate(program, folder, render, views)
                if not gate.success:
                    raise RuntimeError('Controlled initial does not render; no paired model calls sent')
            if file_sha(preflight[0] / 'render.png') != file_sha(preflight[1] / 'render.png'):
                raise RuntimeError('Nondeterministic initial rendering; no paired model calls sent')
            arms = {}
            # Counterbalance serial order; there is no shared conversation or browser session.
            order = ('active', 'fixed') if index % 2 == 0 else ('fixed', 'active')
            for mode in order:
                cfg = IterativeConfig(max_runtime_repairs=runtime_repairs, max_visual_revisions=revisions,
                    verifier=VerifyConfig(budget=budget, mode=mode, exact_budget=True), task=task)
                arms[mode] = run_iterative((reference,), case_dir / mode, render, views, cfg,
                    api.scoped(f'{case_id}/{mode}/generator'), initial_program=program,
                    judge_call=api.scoped(f'{case_id}/{mode}/judge'), judge_model=api.config)
                write_json_atomic(case_dir / 'pair.json', {'order': order, 'arms': arms})
                if arms[mode]['status'] in {'generation_failed', 'verification_error', 'render_failed'}:
                    raise RuntimeError(f'{case_id}/{mode} failed; stop without paid retry')
            if any(file_sha(render.harness_dir / name) != digest for name, digest in report['runtime_files'].items()):
                raise RuntimeError('Harness build changed during a paired run')
            # These gate images never enter either verifier; compare fresh initial browser states.
            gates = [case_dir / mode / 'round-000/render-gate/attempt-000/render.png' for mode in ('active', 'fixed')]
            inputs_equal = (file_sha(gates[0]) == file_sha(gates[1]) and all(
                (case_dir / mode / 'initial.ts').read_text() == program.source for mode in ('active', 'fixed')))
            if not inputs_equal:
                raise RuntimeError('Pair initial source/render mismatch')
            report['pairs'][case_id] = {'order': order, 'initial_inputs_equal': inputs_equal,
                'initial_render_sha': file_sha(gates[0]),
                'arms': {mode: {'status': arm['status'], 'accepted': arm['accepted'],
                    'model_calls': arm['model_calls'], 'total_tokens': arm['total_tokens'],
                    'rounds': len(arm['rounds'])} for mode, arm in arms.items()}}
            write_json_atomic(output / 'result.json', report)
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='incomplete', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['finished_at'] = now_iso()
        write_json_atomic(output / 'result.json', report)
    return report
