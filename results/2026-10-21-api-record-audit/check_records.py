"""Read-only local audit: replay original actions, bind wire inputs, and rescore."""
import base64
import hashlib
import io
import json
from pathlib import Path

from turnitover.api_pipeline.fixture import controlled_cases, reference_program
from turnitover.api_pipeline.scoring import TARGETS, diagnosis_metrics, quality, replay_witnesses
from turnitover.core.actions import Action, action_cost
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import from_dict
from turnitover.models.commands import extract_program
from turnitover.models.streaming import read_chat_stream
from turnitover.policy.loop import execute_observation
from turnitover.render.session import ObservationSession, RenderConfig
from turnitover.render.views import load_views
from turnitover.repair.patches import apply_response
from turnitover.repair.loop import PROTOCOL
from turnitover.verifier.contracts import parse_json

ROOT = Path.cwd()
OUT = ROOT / 'results/2026-10-21-api-record-audit'
rows, arms, frames = [], [], []
read = lambda p: json.loads(Path(p).read_text())
sha = lambda b: hashlib.sha256(b).hexdigest()


def check(name, value):
    rows.append({'check': name, 'passed': bool(value)})
    assert value, name


def bind(folder, prompt, images, response=None):
    wire = read(folder / 'request.json')
    content = wire['messages'][0]['content']
    check(str(folder.relative_to(ROOT)) + ':prompt', [c['text'] for c in content if c['type'] == 'text'] == [prompt])
    actual = [base64.b64decode(c['image_url']['url'].split(',', 1)[1]) for c in content if c['type'] == 'image_url']
    check(str(folder.relative_to(ROOT)) + ':images', actual == [p.read_bytes() for p in images])
    if response is not None:
        check(str(folder.relative_to(ROOT)) + ':response', read(folder / 'response.json')['choices'][0]['message']['content'] == response)
    if (folder / 'response.sse').exists():
        check(str(folder.relative_to(ROOT)) + ':SSE', read_chat_stream(io.BytesIO((folder / 'response.sse').read_bytes())) == read(folder / 'response.json'))
    return wire


render = RenderConfig(ROOT / 'web/dist', ROOT / 'web/node_modules/.bin/esbuild', width=256, height=256)
views = load_views(ROOT / 'configs/views.yaml')
prepared = ROOT / 'output/api-prepared-v2'
base = ObjectProgram((prepared / 'shared/final.ts').read_text())
reference = ObjectProgram((prepared / 'private/reference.ts').read_text())
with ObservationSession(render, views) as session:
    session.load(reference, framing=None)
    check('reference_png_exact_chromium_render', session.request_view('oblique_fr').png == (prepared / 'reference.png').read_bytes())
check('reference_authored_separately', reference.source == reference_program().source and reference.source != base.source)
generation = prepared / 'shared/generation'
gen_api = ROOT / 'output/api-prepared-v2-api/api_calls/001'
raw = read(gen_api / 'response.json')['choices'][0]['message']['content']
check('generated_base_exact_extraction', extract_program(raw) == base.source)
bind(gen_api, (generation / 'prompt.txt').read_text(), [generation / r['path'] for r in read(generation / 'result.json')['images']], raw)
cases = controlled_cases(base)
cache = {}
for run_name in ['api-minimal-v6', 'api-paired-v6']:
    run = ROOT / 'output' / run_name
    manifest = read(run / 'result.json')
    scores = read(run / 'scores.private.json')
    ledgers = [ROOT / 'output/api-minimal-v5-api', ROOT / 'output/api-minimal-v6-api'] if run_name == 'api-minimal-v6' else [ROOT / 'output/api-paired-v6-api']
    calls = [(path / c['directory'], c) for path in ledgers for c in read(path / 'ledger.json')['calls']]
    for case, pair in manifest['pairs'].items():
        check(case + ':controlled_source', (run / case / 'initial.ts').read_text() == cases[case].source)
        gates = [run / case / mode / 'round-000/render-gate/attempt-000/render.png' for mode in ['active', 'fixed']]
        check(case + ':initial_render_equal', gates[0].read_bytes() == gates[1].read_bytes())
        for mode in ['active', 'fixed']:
            arm = run / case / mode
            saved = scores['pairs'][case][mode]
            judge_calls = iter([(p, c) for p, c in calls if c['scope'].startswith(case + '/' + mode + '/judge')])
            for rd in saved['rounds']:
                folder = arm / f"round-{rd['round']:03d}/verifier"
                verdict = read(folder / 'result.json')
                history = [json.loads(line) for line in (folder / 'trajectory.jsonl').read_text().splitlines()]
                check(str(folder.relative_to(ROOT)) + ':budget', verdict['spent'] == sum(action_cost(from_dict(Action, o['action'])) for o in history) == len(history) == 4)
                check(str(folder.relative_to(ROOT)) + ':trajectory', history == verdict['trajectory'])
                program = ObjectProgram((folder / 'candidate.ts').read_text())
                with ObservationSession(render, views) as session:
                    session.load(program, framing=None)
                    for o in history:
                        acquired = []
                        fresh = execute_observation(session, from_dict(Action, o['action']), o['step'], lambda step, png: acquired.append(png) or 'replayed')
                        if o['image_ref']:
                            original = (folder / o['image_ref']).read_bytes()
                            equal = len(acquired) == 1 and acquired[0] == original
                            frames.append({'path': str((folder / o['image_ref']).relative_to(ROOT)), 'sha256': sha(original), 'replay_byte_equal': equal})
                            check(str(folder.relative_to(ROOT)) + ':frame:' + str(o['step']), equal)
                        else:
                            check(str(folder.relative_to(ROOT)) + ':runtime:' + str(o['step']), {k:v for k,v in fresh.payload.items() if k != 'last_render_ms'} == {k:v for k,v in o['payload'].items() if k != 'last_render_ms'})
                if program.sha not in cache:
                    cache[program.sha] = quality(program, reference, render, views)
                check(str(folder.relative_to(ROOT)) + ':quality', cache[program.sha] == rd['quality'])
                if rd['round'] == 0:
                    witnesses = replay_witnesses(program, reference, history, render, views)
                    check(case + '/' + mode + ':witnesses', witnesses == rd['witnesses'])
                    check(case + '/' + mode + ':diagnosis', diagnosis_metrics(verdict, TARGETS[case], witnesses) == rd['diagnosis'])
                for item in verdict['model_calls']:
                    local = folder / item['path']
                    prompt = (local / 'prompt.txt').read_text()
                    data = json.JSONDecoder().raw_decode(prompt.split('\nObserved task data:\n', 1)[1])[0]
                    check(str(local.relative_to(ROOT)) + ':causal_history', data['history'] == history[:len(data['history'])] and data['remaining'] == 4 - len(data['history']))
                    images = [folder / name for name in read(local / 'images.json')]
                    check(str(local.relative_to(ROOT)) + ':image_order', images == [folder / r['path'] for r in verdict['references']] + [folder / o['image_ref'] for o in data['history'] if o['image_ref']])
                    path, entry = next(judge_calls)
                    bind(path, prompt, images, (local / 'response.txt').read_text() if item['status'] != 'failed' else None)
                    check(str(path.relative_to(ROOT)) + ':frozen_request', all(read(path / 'request.json')[k] == v for k, v in {'model':'k3-256k', 'temperature':1, 'max_tokens':8192}.items()))
            check(case + '/' + mode + ':all_judge_calls_bound', next(judge_calls, None) is None)
            final = ObjectProgram((arm / 'final.ts').read_text())
            check(case + '/' + mode + ':final_quality', quality(final, reference, render, views) == saved['final_quality'])
            arms.append({'case':case, 'mode':mode, 'final_correct':saved['final_quality']['code_correct'], 'pass_count':saved['final_quality']['invariant_pass_count'], 'count':saved['final_quality']['invariant_count']})
            for path, entry in calls:
                if entry['scope'] != case + '/' + mode + '/generator': continue
                repairs = list(arm.glob('round-*/visual-revisions/revision-*'))
                check(case + '/' + mode + ':single_repair', len(repairs) == 1)
                repair = repairs[0]
                prompt = (repair / 'prompt.txt').read_text()
                data = json.loads(prompt.removeprefix(PROTOCOL + '\n'))
                images = [repair / name for name in data['image_order']]
                bind(path, prompt, images, (repair / 'response.txt').read_text())
                before = (repair.parents[1] / 'candidate.ts').read_text()
                check(case + '/' + mode + ':repair_current_source', before == data['current_program'])
                after = apply_response(before, parse_json((repair / 'response.txt').read_text()))
                check(case + '/' + mode + ':exact_patch_final', final.source == (after if after is not None else before))
old = ROOT / 'output/api-minimal-v5'
new = ROOT / 'output/api-minimal-v6'
continuation = read(new / 'result.json')['continuation']
check('continuation_source_hashes', all(sha((old / name).read_bytes()) == digest for name, digest in continuation['copied_history_sha256'].items()))
check('continuation_observations_unchanged', all(p.read_bytes() == (new / p.relative_to(old)).read_bytes() for pattern in ['*.png', '*.ts', 'trajectory.jsonl'] for p in old.rglob(pattern)))
a = read(ROOT / 'output/api-minimal-v5-api/api_calls/011/request.json')
b = read(ROOT / 'output/api-minimal-v6-api/api_calls/001/request.json')
check('continuation_request_equal_except_stream', {k:v for k,v in a.items() if k not in ['stream','stream_options']} == {k:v for k,v in b.items() if k not in ['stream','stream_options']})
(OUT / 'checks.json').write_text(json.dumps({'status':'PASS', 'api_requests':0, 'checks':rows, 'frames':frames, 'arms':arms}, indent=2) + '\n')
print(json.dumps({'checks':len(rows), 'checks_passed':all(r['passed'] for r in rows), 'replayed_images':len(frames), 'image_byte_matches':sum(r['replay_byte_equal'] for r in frames), 'arms':arms}, indent=2))
