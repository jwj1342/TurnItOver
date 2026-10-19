"""Verify retained real artifacts and recompute report counts without model calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output/partnet-action-sft-v1'
EVIDENCE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    from turnitover.core.serde import to_dict
    from turnitover.verifier.contracts import Context, parse_decision, parse_json
    comparison = read(OUT/'comparison.json')
    training = read(OUT/'train/run.json')
    loss = [json.loads(s) for s in (OUT/'train/loss.jsonl').read_text().splitlines()]
    assert training['status'] == 'complete' and training['steps'] == len(loss) == 1488
    assert Counter(r['epoch'] for r in loss) == {0:496, 1:496, 2:496}
    from random import Random
    from scripts.evaluate_action_sft_learning import target_mass
    grouped = {}
    prefixes = {}
    for split in ('train','validation'):
        for line in (ROOT/f'data/partnet-action-distributions-v1/{split}.actions.jsonl').read_text().splitlines():
            record = json.loads(line)
            key = record['id'].rsplit(':',1)[0]
            grouped.setdefault(key, []).append((record, []))
            prefixes[key] = json.loads(record['messages'][0]['content'].rsplit('\n',1)[1])
    train_groups = [value for key,value in grouped.items() if key.startswith('train:')]
    for epoch in range(3):
        expected = list(range(496))
        Random(training['seed']+epoch).shuffle(expected)
        assert [r['group'] for r in loss if r['epoch']==epoch] == expected
    for update in loss:
        variants = train_groups[update['group']]
        assert [v['id'] for v in update['variants']] == [r['id'] for r,_ in variants]
        assert [v['weight'] for v in update['variants']] == [r['loss_weight'] for r,_ in variants]
        assert abs(sum(v['weight']*v['loss'] for v in update['variants'])-update['loss']) < 1e-12
    learn_rows = [json.loads(s) for s in (OUT/'learning/groups.jsonl').read_text().splitlines()]
    for row in learn_rows:
        mass = 0.
        if row['legal']:
            assert row['generation']['finish_reason'] == 'completed'
            mass = target_mass(row['generation']['text'], grouped[row['group']])
        assert abs(row['empirical_target_mass']-mass) < 1e-12
    for key, summary in read(OUT/'learning/metrics.json')['summary'].items():
        split,mode = key.split('/')
        rows = [r for r in learn_rows if r['split']==split and r['mode']==mode]
        assert len(rows) == summary['groups']
        for metric in ('weighted_assistant_nll','legal','empirical_target_mass','supported_action','majority_mass'):
            assert abs(sum(r[metric] for r in rows)/len(rows)-summary[metric]) < 1e-12
    checkpoint = OUT/'train/checkpoint/adapter_model.safetensors'
    assert sha(checkpoint) == training['checkpoint_sha256'][checkpoint.name]
    ready = [json.loads(s) for s in (OUT/'sft-server.log').read_text().splitlines()
             if s.startswith('{"status": "ready"')][0]
    assert ready['adapter_loaded'] and ready['adapter_sha256'] == sha(checkpoint)
    assert ready['final_diagnoser'] == 'frozen_base_adapter_disabled'
    image_count = action_count = episode_count = 0
    for name in ('baselines-run3', 'action-sft'):
        run = OUT/name
        metrics = read(run/'metrics.json')
        assert metrics['status'] == 'complete' and metrics['score_eligible']
        for relative, digest in read(run/'images.sha256.json').items():
            assert sha(run/relative) == digest, relative
            image_count += 1
        for row in metrics['episodes']:
            folder = run/row['folder']
            episode = read(folder/'episode.json')
            assert episode['spent'] == row['spent'] == row['budget']
            assert len(episode['trajectory']) == row['legal_actions'] == row['budget']
            assert row['action_attempts'] == row['legal_actions']
            if row['mode'] in ('zero_shot', 'action_sft'):
                for step, observation in enumerate(episode['trajectory']):
                    call = folder/'action_calls'/f'{step:03d}'
                    payload = json.loads((call/'prompt.txt').read_text().rsplit('\n',1)[1])
                    assert len(payload['history']) == step
                    assert payload['remaining'] == row['budget']-step
                    assert set(payload) == {'context','remaining','history','image_order','taxonomy_ids'}
                    assert payload['image_order'] == read(call/'images.json')
                    assert len(payload['image_order']) == step+1
                    assert payload['history'] == [{
                        'step':o['step'], 'action':parse_json((folder/'action_calls'/f'{i:03d}'/'response.txt').read_text())['action'],
                        'image_ref':o['image_ref'], 'payload':{k:v for k,v in o['payload'].items() if k!='render_ms'}}
                        for i,o in enumerate(episode['trajectory'][:step])]
                    parsed = parse_decision(parse_json((call/'response.txt').read_text()),
                                            Context(**payload['context']), set(range(step)))
                    assert to_dict(parsed) == observation['action']
                    assert read(call/'metadata.json')['legal_action']
                    action_count += 1
            private = read(folder/'evaluation.private.json')
            assert private['exposed'] == row['exposed']
            assert private['diagnostic_exact'] == row['diagnostic_exact']
            episode_count += 1
    assert episode_count == 336 and action_count == 672
    table = {}
    for key, summary in comparison['summary'].items():
        budget, mode = key.split('/')
        rows = [r for r in comparison['episodes'] if r['mode']==mode and r['budget']==int(budget[1:])]
        assert len(rows)==28 and len({r['asset_id'] for r in rows})==7
        assert sum(r['labels'] for r in rows)==29
        assert summary['exposure_recall'] == sum(r['exposed'] for r in rows)/29
        assert summary['diagnosis_accuracy'] == sum(r['diagnostic_exact'] for r in rows)/28
        table[key] = dict(exposed=sum(r['exposed'] for r in rows), labels=29,
            correct=sum(r['diagnostic_exact'] for r in rows), cases=28,
            valid_final=sum(r['final_readout']['status']=='complete' for r in rows),
            observations=sum(r['spent'] for r in rows))
    result = dict(status='pass', episodes=episode_count, active_actions=action_count,
                  image_files=image_count, adapter_sha256=sha(checkpoint), counts=table)
    (EVIDENCE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
