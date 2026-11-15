"""Frozen Qwen diagnosis on identical acquired trajectories, single vs paired references.

This evaluates recognition given saved evidence, not learned action selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from turnitover.dataset.metrics import count_labels, detection_metrics
from turnitover.models.local_qwen import LocalQwen
from turnitover.oracle.readout import load_readout_data
from turnitover.taxonomy import load_taxonomy
from turnitover.verifier.contracts import Context, parse_json, parse_decision


def run(oracle, paired, model_path, output, limit=None, final_protocol=False):
    oracle_manifest, pairs, annotations = load_readout_data(oracle, paired)
    benchmark = Path(oracle_manifest['benchmark'])
    inputs = {r['id']: r for r in map(json.loads, (benchmark / 'prepared/test.inputs.jsonl').read_text().splitlines())}
    gold = {r['id']: r for r in map(json.loads, (benchmark / 'prepared/test.gold.jsonl').read_text().splitlines())}
    if limit is not None:
        annotations = annotations[:limit]
    if not annotations:
        raise ValueError('No test examples')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='incomplete', model=str(model_path), samples=len(annotations),
                  prompt_variant='final_only_contract' if final_protocol else 'action_and_verdict_contract',
                  scope='frozen_VLM_recognition_on_saved_trajectories',
                  limitations=['Single and paired conditions differ in reference information.',
                               'Oracle trajectories use privileged synthetic truth.',
                               'Invalid outputs count as missed detections, not successful uncertainty.',
                               'Six asset families; the test split contains only the door family.',
                               'No active model action selection or repair loop evaluated.'])
    model = LocalQwen(model_path)
    metrics = {}
    protocol = (Path(__file__).resolve().parents[1] / 'turnitover/verifier/prompt.txt').read_text()
    final_text = (Path(__file__).resolve().parents[1] / 'turnitover/verifier/final_prompt.txt').read_text()
    try:
        with (output / 'predictions.jsonl').open('w') as stream:
            for annotation in annotations:
                uid = annotation['id']
                context = inputs[uid]['context']
                for mode, trajectory in annotation['trajectories'].items():
                    history = trajectory['history']
                    for condition in ('single', 'paired'):
                        key = f'{mode}/{condition}'
                        metric = metrics.setdefault(key, dict(
                            n=0, invalid=0, statuses=Counter(),
                            classes={label: Counter() for label in load_taxonomy().implemented_ids()}))
                        if condition == 'single':
                            images = [Path('images') / uid / 'reference.png'] + [Path(o['image_ref']) for o in history]
                            prompt = protocol + '\n' + json.dumps(dict(context=context, remaining=0, history=history,
                                taxonomy_ids=load_taxonomy().ids, image_order=[str(p) for p in images]))
                            paths = [oracle / p for p in images]
                        else:
                            row = pairs[f'{uid}:{mode}:paired']
                            prompt = row['messages'][0]['content']
                            paths = [paired / p for p in row['images']]
                        if final_protocol:
                            if not prompt.startswith(protocol):
                                raise ValueError('Unexpected source prompt; cannot replace protocol')
                            prompt = final_text + prompt[len(protocol):]
                        # Only public prompt/images enter the model; full gold is scored afterwards.
                        response = model(prompt, paths)
                        decision, error, predicted = None, None, set()
                        try:
                            if response.finish_reason != 'completed':
                                raise ValueError('Generation incomplete')
                            decision = parse_json(response.text)
                            verdict = parse_decision(decision, Context(**context), {o['step'] for o in history}, final_only=True)
                            metric['statuses'][verdict.status] += 1
                            predicted = {f.defect_id for f in verdict.findings}
                        except ValueError as exc:
                            error = str(exc)
                            metric['invalid'] += 1
                        expected = {f['defect_id'] for f in gold[uid]['target']['findings']}
                        count_labels(metric['classes'], expected, predicted)
                        metric['n'] += 1
                        stream.write(json.dumps(dict(id=uid, mode=mode, condition=condition, raw=response.text,
                            decision=decision, error=error, expected=sorted(expected), predicted=sorted(predicted)))+'\n')
                        stream.flush()
                        print(json.dumps(dict(id=uid, condition=key, error=error)), flush=True)
        for metric in metrics.values():
            metric['classes'] = detection_metrics(metric['classes'])
        report.update(status='complete', metrics=metrics)
    except Exception as exc:
        report.update(error=str(exc), metrics=metrics)
        raise
    finally:
        (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, default=Path('data/benchmarks/oracle-v1'))
    parser.add_argument('--paired', type=Path, default=Path('data/benchmarks/paired-v1'))
    parser.add_argument('--model', type=Path, default=Path('models/Qwen3-VL-2B-Instruct'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--final-protocol', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.oracle, args.paired, args.model, args.out, args.limit, args.final_protocol), indent=2))
