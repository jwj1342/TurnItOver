"""Read-only raw-record checks; writes only a new review trace, never runs an API."""
from pathlib import Path
import base64
import collections
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
hashes, checks = {}, []


def read(path):
    p = ROOT / path
    raw = p.read_bytes()
    hashes[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw


def js(path):
    return json.loads(read(path))


def sha(path):
    return hashlib.sha256(read(path)).hexdigest()


def check(name, passed, detail=None):
    checks.append(dict(name=name, passed=bool(passed), detail=detail))


summary = js('results/2026-10-21-api-pipeline/summary.json')
ledger_names = sorted({r['ledger'] for r in summary['calls']})
all_calls, wire_records = [], {}
for ledger_name in ledger_names:
    ledger = js(ledger_name)
    folder = Path(ledger_name).parent
    check(f'{folder}: count', len(ledger['calls']) == ledger['request_count'])
    for entry in ledger['calls']:
        directory = folder / entry['directory']
        request = js(directory / 'request.json')
        prompt = read(directory / 'prompt.txt').decode()
        images = js(directory / 'images.json')
        content = request['messages'][0]['content']
        wire_images = [hashlib.sha256(base64.b64decode(x['image_url']['url'].split(',', 1)[1])).hexdigest()
                       for x in content if x['type'] == 'image_url']
        check(f'{directory}: exact text', [x['text'] for x in content if x['type'] == 'text'] == [prompt])
        check(f'{directory}: exact images', wire_images == [x['sha256'] for x in images]
              == [sha(directory / x['name']) for x in images])
        check(f'{directory}: request config', request['model'] == ledger['config']['model']
              and request['max_tokens'] == ledger['config']['max_tokens']
              and request.get('temperature') == ledger['config'].get('temperature', request.get('temperature')))
        metadata = js(directory / 'metadata.json')
        check(f'{directory}: ledger usage', metadata['usage'] == entry['usage'])
        response = None
        if (ROOT / directory / 'response.json').exists():
            response = js(directory / 'response.json')
            check(f'{directory}: raw usage', response.get('usage', {}) == entry['usage'])
        if (ROOT / directory / 'response.sse').exists():
            raw = read(directory / 'response.sse').decode()
            events = [event for event in re.split(r'\r?\n\r?\n', raw) if event.strip()]
            payloads = ['\n'.join(line[5:].lstrip(' ') for line in e.splitlines() if line.startswith('data:')) for e in events]
            payloads = [p for p in payloads if p]
            check(f'{directory}: SSE terminal', payloads[-1] == '[DONE]')
            chunks = [json.loads(p) for p in payloads if p != '[DONE]']
            choices = [c for chunk in chunks for c in chunk.get('choices', [])]
            assembled = ''.join(c.get('delta', {}).get('content') or '' for c in choices)
            reasons = [c['finish_reason'] for c in choices if c.get('finish_reason')]
            usage = [c['usage'] for c in chunks if c.get('usage') is not None]
            models = {c['model'] for c in chunks if c.get('model')}
            check(f'{directory}: independent SSE assembly', assembled == response['choices'][0]['message']['content']
                  and reasons == [response['choices'][0]['finish_reason']]
                  and usage[-1] == response['usage'] and models == {response['model']})
        if entry['status'] == 'complete':
            check(f'{directory}: complete response', response is not None
                  and response['choices'][0]['message']['content'].strip()
                  and response['choices'][0]['finish_reason'] == 'stop'
                  and response['model'] == 'k3-256k')
        else:
            check(f'{directory}: failure retained', (ROOT / directory / 'error.json').exists())
        wire_records[str(directory)] = dict(prompt=prompt, image_hashes=wire_images, response=response)
        all_calls.append(dict(ledger=ledger_name, **entry))

totals = dict(request_attempts=len(all_calls), request_status=dict(collections.Counter(c['status'] for c in all_calls)),
              reported_tokens=sum(c['usage'].get('total_tokens', 0) for c in all_calls),
              attempts_without_usage=sum(not c['usage'] for c in all_calls),
              reported_cost_count=sum(c['provider_cost'] is not None for c in all_calls))
for key, value in totals.items():
    check(f'summary {key}', value == summary['api_accounting'][key], value)
check('summary no duplicated ledger/index', len(all_calls) == len({(r['ledger'], r['index']) for r in all_calls}))
check('summary all original entries exactly once', all_calls == sorted(summary['calls'], key=lambda r: (r['ledger'], r['index'])))

prepared = Path('output/api-prepared-v2')
manifest = js(prepared / 'manifest.json')
check('prepared file manifest', all(sha(prepared / p) == digest for p, digest in manifest['files'].items()))
base = read(prepared / 'shared/final.ts').decode()
reference = read(prepared / 'private/reference.ts').decode()
gen = wire_records['output/api-prepared-v2-api/api_calls/001']
gen_text = gen['response']['choices'][0]['message']['content']
extracted = re.search(r'```(?:typescript|ts)\s*\n(.*?)```', gen_text, re.S).group(1).strip() + '\n'
check('generation content bound to final source', extracted == base)
check('base differs from authored reference', base != reference)
check('generation request actual reference image', gen['image_hashes'] == [sha(prepared / 'reference.png')])
check('generation exact local request', gen['prompt'] == read(prepared / 'shared/generation/prompt.txt').decode()
      and gen_text == read(prepared / 'shared/generation/response.txt').decode())
check('private reference absent from all prompts', all(reference not in x['prompt'] for x in wire_records.values()))
check('no explicit private labels/checker payload in calls', all(not any(s in x['prompt'] for s in
      ['invariant_pass_count', 'scores.private.json', 'base_quality', 'mismatched_parts', 'private/reference.ts']) for x in wire_records.values()))

rows = []
for run_name in ['api-minimal-v6', 'api-paired-v6']:
    run = Path('output') / run_name
    report = js(run / 'result.json')
    scores = js(run / 'scores.private.json')
    for case_id in report['pairs']:
        initial = read(run / case_id / 'initial.ts').decode()
        check(f'{run}/{case_id}: initial hash', hashlib.sha256(initial.encode()).hexdigest()[:16] == report['initials'][case_id])
        if case_id == 'c00':
            check('clean initial == API base', initial == base)
        else:
            prefix = re.sub(r'export\s+default\s+function\s+createObject\b', 'function createInitialObject', base)
            check(f'{case_id}: injected candidate retains API body', initial.startswith(prefix))
        for mode in ['active', 'fixed']:
            arm = run / case_id / mode
            result = js(arm / 'result.json')
            check(f'{arm}: source equals paired initial', read(arm / 'initial.ts').decode() == initial)
            final = read(arm / 'final.ts').decode()
            score = scores['pairs'][case_id][mode]
            obs_count, judge_count, image_counts = 0, 0, []
            for rr in result['rounds']:
                verifier = arm / f"round-{rr['round']:03d}/verifier"
                v = js(verifier / 'result.json')
                trajectory = [json.loads(x) for x in read(verifier / 'trajectory.jsonl').decode().splitlines()]
                check(f'{verifier}: exact persisted history', trajectory == v['trajectory'])
                check(f'{verifier}: K=4', v['spent'] == 4 == len(trajectory))
                check(f'{verifier}: candidate sha', sha(verifier / 'candidate.ts')[:16] == v['program_sha'])
                check(f'{verifier}: refs actual prepared reference', all(sha(verifier / x['path']) == sha(prepared / 'reference.png') for x in v['references']))
                obs_count += v['spent']; judge_count += len(v['model_calls'])
                image_counts.append(sum(bool(o.get('image_ref')) for o in trajectory))
                for mc in v['model_calls']:
                    call_dir = verifier / mc['path']
                    p = read(call_dir / 'prompt.txt').decode()
                    local_images = [sha(verifier / name) for name in js(call_dir / 'images.json')]
                    matches = [(key, w) for key, w in wire_records.items() if p == w['prompt'] and local_images == w['image_hashes']]
                    check(f'{call_dir}: exact actual wire binding', bool(matches), [m[0] for m in matches])
                    if (ROOT / call_dir / 'response.txt').exists():
                        response_text = read(call_dir / 'response.txt').decode()
                        check(f'{call_dir}: raw response binding', any(w['response'] and w['response']['choices'][0]['message']['content'] == response_text for _, w in matches))
                    data = json.loads(p.split('\nObserved task data:\n', 1)[1].split('\nThis run requires', 1)[0])
                    check(f'{call_dir}: public context equality', data['context'] == v['context'])
                    prefix_len = 4 - data['remaining']
                    check(f'{call_dir}: only actual prefix observations', data['history'] == trajectory[:prefix_len])
                    expected = [sha(verifier / x['path']) for x in v['references']]
                    expected += [sha(verifier / o['image_ref']) for o in trajectory[:prefix_len] if o.get('image_ref')]
                    check(f'{call_dir}: exact permitted image order', local_images == expected)
                q = score['rounds'][rr['round']]['quality']
                invariants = [i for c in q['checks'] for i in c.get('invariants', [])]
                check(f'{verifier}: raw invariants denominator', len(invariants) == q['invariant_count'] == 32
                      and sum(i['passed'] for i in invariants) == q['invariant_pass_count'])
            related = [c for c in all_calls if c['scope'].startswith(f'{case_id}/{mode}/') and
                       c['ledger'] in (['output/api-minimal-v5-api/ledger.json', 'output/api-minimal-v6-api/ledger.json']
                       if run_name == 'api-minimal-v6' else ['output/api-paired-v6-api/ledger.json'])]
            check(f'{arm}: judge counter', judge_count == result['model_calls']['judge'])
            check(f'{arm}: API counters', len(related) == score['api_cost']['requests']
                  and sum(c['usage'].get('total_tokens', 0) for c in related) == result['total_tokens'] == score['api_cost']['tokens'])
            check(f'{arm}: observation counters', obs_count == score['total_observations'])
            check(f'{arm}: final is last verifier candidate', final == read(arm / f"round-{result['rounds'][-1]['round']:03d}/verifier/candidate.ts").decode())
            first = js(arm / 'round-000/verifier/result.json')
            expected = {'c00': set(), 'c01': {'structure.part_offset'}, 'c02': {'kinematics.joint_axis'}}[case_id]
            predicted = {f['defect_id'] for f in first['verdict']['findings']}
            expected_status = 'fail' if expected else 'pass'
            check(f'{arm}: diagnostic counts', score['rounds'][0]['diagnosis']['detection_correct'] == (first['verdict']['status'] == expected_status)
                  and score['rounds'][0]['diagnosis']['type_exact'] == (expected == predicted))
            rows.append(dict(case_id=case_id, mode=mode, observations=obs_count, observation_images=image_counts,
                             requests=len(related), tokens=result['total_tokens'], verdict=first['verdict']['status'],
                             final_quality=f"{score['final_quality']['invariant_pass_count']}/32"))

for repair_name, wire_name, source_name in [
    ('output/api-minimal-v6/c02/active/round-000/visual-revisions/revision-000',
     'output/api-minimal-v5-api/api_calls/006', 'output/api-minimal-v6/c02/active/initial.ts'),
    ('output/api-paired-v6/c00/fixed/round-000/visual-revisions/revision-000',
     'output/api-paired-v6-api/api_calls/007', 'output/api-paired-v6/c00/fixed/initial.ts')]:
    repair_dir = Path(repair_name)
    prompt = read(repair_dir / 'prompt.txt').decode()
    payload = json.loads(prompt[prompt.index('{"current_program"'):])
    wire = wire_records[wire_name]
    check(f'{repair_name}: generator wire matches local evidence', prompt == wire['prompt'] and
          [sha(repair_dir / name) for name in payload['image_order']] == wire['image_hashes'] and
          read(repair_dir / 'response.txt').decode() == wire['response']['choices'][0]['message']['content'])
    check(f'{repair_name}: actual current candidate', payload['current_program'] == read(source_name).decode())
    check(f'{repair_name}: no private GT feedback', payload['feedback'] == js(repair_dir / 'feedback.json') and
          payload['feedback']['task'] == read(prepared / 'task.txt').decode() and payload['previous_attempts'] == [])

old = Path('output/api-minimal-v5')
new = Path('output/api-minimal-v6')
continuation = js(new / 'result.json')['continuation']
changed = []
for path, digest in continuation['copied_history_sha256'].items():
    check(f'recovery source immutable: {path}', sha(old / path) == digest)
    if sha(new / path) != digest:
        changed.append(path)
req_old = js('output/api-minimal-v5-api/api_calls/011/request.json')
req_new = js('output/api-minimal-v6-api/api_calls/001/request.json')
check('recovery exact wire except transport', req_old == {k: v for k, v in req_new.items() if k not in {'stream', 'stream_options'}})
check('recovery source config', js(old / 'result.json')['model'] == js(new / 'result.json')['model'])
check('recovery source hash recorded', sha(old / 'result.json') == continuation['source_result_sha256'])
check('recovery no additional active observation image', set(p.relative_to(ROOT / old / 'c02/active') for p in (ROOT / old / 'c02/active').rglob('observations/*.png')) ==
      set(p.relative_to(ROOT / new / 'c02/active') for p in (ROOT / new / 'c02/active').rglob('observations/*.png')))

repair = Path('output/api-minimal-v6/c02/active/round-000/visual-revisions/revision-000')
repair_text = read(repair / 'response.txt').decode().strip()
response = json.loads(repair_text[8:-3].strip() if repair_text.startswith('```json\n') else repair_text)
patched = read(new / 'c02/active/initial.ts').decode()
for edit in response['edits']:
    check('repair target unique', patched.count(edit['old']) == 1)
    patched = patched.replace(edit['old'], edit['new'])
check('repair exact patch output and final', patched == read(repair / 'proposed.ts').decode() == read(new / 'c02/active/final.ts').decode())
check('false-positive repair returns stop', js('output/api-paired-v6/c00/fixed/round-000/visual-revisions/revision-000/response.txt') == {'stop': True})

result = dict(totals=totals, rows=rows, recovery_changed_files=changed, checks=checks,
              failures=[c for c in checks if not c['passed']], input_hashes=hashes)
(TRACE / 'record-checks.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: result[k] for k in ['totals', 'rows', 'recovery_changed_files', 'failures']}, indent=2))
