#!/usr/bin/env python3
"""Verify fixed-budget artifacts and paired predictions before interpreting this pilot."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(root, report):
    data = json.loads((report/'data-audit.json').read_text())
    model = json.loads((report/'model-manifest.json').read_text())
    kimi_ids = {row['id']:row['category'] for row in data['rows']}
    result = {'status':'passed', 'arms':{}, 'test_used':False, 'selection':'final step only; no Validation checkpoint selection'}
    runs = []
    for arm in ('baseline','experiment'):
        training = root/f'{arm}-train'
        evaluation = root/f'{arm}-eval'
        run = json.loads((training/'run.json').read_text())
        metrics = json.loads((evaluation/'metrics.json').read_text())
        assert run['status'] == metrics['status'] == 'complete'
        assert run['steps'] == run['max_steps'] == 1617
        assert run['objectives']['counts'] == {'diagnosis':110 if arm == 'baseline' else 120, 'stopping':429}
        expected_episodes = data['baseline_input_hashes']['train.episodes.jsonl'] if arm == 'baseline' else sha(root.parent/'experiment-data/train.episodes.jsonl')
        assert run['train_episodes_sha256'] == expected_episodes
        assert run['dataset_manifest_sha256'] == data['baseline_input_hashes']['manifest.json']
        assert run['seed'] == 20261013 and run['epochs'] == 3
        assert run['learning_rate'] == 2e-4 and run['supervision'] == 'evidence'
        assert run['lora'] == {'rank':8, 'alpha':16, 'dropout':0., 'target_modules':['q_proj','k_proj','v_proj','o_proj']}
        assert run['objectives']['loss_weights'] == {'diagnosis':1., 'stopping':110/429}
        assert run['objectives']['supervision_sha256'] == data[f'{arm}_supervision_sha256']
        assert run['base_model_config_sha256'] == model['config.json']['sha256']
        assert run['train_stopping_sha256'] == data['baseline_input_hashes']['train.stopping.jsonl']
        assert metrics['validation_episodes_sha256'] == data['baseline_input_hashes']['validation.episodes.jsonl']
        assert metrics['dataset_manifest_sha256'] == data['baseline_input_hashes']['manifest.json']
        assert metrics['n'] == 28 and metrics['max_new_tokens'] == 512
        losses = read(training/'loss.jsonl')
        planned = read(report/f'{arm}-planned-order.jsonl')
        assert len(losses) == len(planned) == 1617
        for actual, expected in zip(losses, planned, strict=True):
            assert all(actual[key] == expected[key] for key in ('step','epoch','id','objective'))
            assert all(math.isfinite(actual[key]) and actual[key] >= 0 for key in ('raw_loss','weighted_loss'))
        result['arms'][arm] = {'steps':len(losses), 'objective_steps':dict(Counter(r['objective'] for r in losses)),
                              'checkpoint_sha256':sha(training/'checkpoint/adapter_model.safetensors'),
                              'train_run_sha256':sha(training/'run.json'), 'loss_sha256':sha(training/'loss.jsonl'),
                              'predictions_sha256':sha(evaluation/'predictions.jsonl'),
                              'metrics_sha256':sha(evaluation/'metrics.json'),
                              'actual_order_matches_plan':True,
                              'kimi_exposures':sum(r['id'] in kimi_ids for r in losses),
                              'kimi_class_exposures':dict(Counter(kimi_ids[r['id']] for r in losses if r['id'] in kimi_ids))}
        runs.append(run)
    for key in ('base_model', 'seed','learning_rate','lora','python','torch','cuda','gpu','trainable_parameters','total_parameters'):
        assert runs[0][key] == runs[1][key], key
    assert runs[0]['examples'] == 539 and runs[1]['examples'] == 549
    ids = [set(r['id'] for r in read(root/f'{arm}-eval/predictions.jsonl')) for arm in ('baseline','experiment')]
    assert ids[0] == ids[1] and len(ids[0]) == 28
    (report/'run-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    audit(args.root,args.report)
