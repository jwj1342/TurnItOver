"""Recompute v5 public evidence, causal closure, action audit and matched holdout pairs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from turnitover.oracle.supervision import load_examples, action_json
from turnitover.oracle.motion_decisions import schedules, advance
from turnitover.oracle.visual_decisions import BUDGET, VIEWS, ORDERS, image_evidence, make_row, deduplicate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def payload(row):
    return json.loads(row['messages'][0]['content'].rsplit('\n', 1)[-1])


def stage(row):
    p = payload(row)
    return (len(p['history']), p['remaining'], len(row['images']))


def public_key(row):
    p = payload(row)
    # Strip path identifiers, retaining all non-image public information.
    for i, h in enumerate(p['history']):
        h['image_ref'] = f'observation_{i}'
    p['image_order'] = list(range(len(p['image_order'])))
    return json.dumps(p, sort_keys=True)


def stats(rows):
    buckets = defaultdict(Counter)
    conflicts = defaultdict(set)
    for row in rows:
        label = (row['stop_target'], row['stop_outcome'])
        buckets[stage(row)][str(label)] += 1
        conflicts[(public_key(row), tuple(row['image_sha256']))].add(label)
    return {'rows': len(rows), 'objects': len({r['data_identity']['split_group'] for r in rows}),
            'labels': dict(Counter(f"{r['stop_target']}/{r['stop_outcome']}" for r in rows)),
            'stage_label_counts': {str(k): dict(v) for k, v in sorted(buckets.items())},
            'stage_majority_upper_bound': sum(max(v.values()) for v in buckets.values()) / len(rows),
            'identical_public_input_label_conflicts': sum(len(v) > 1 for v in conflicts.values())}


def pair_rows(rows):
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if payload(row)['remaining'] <= 0:
            continue
        label = row['stop_outcome'] or row['stop_target']
        if label not in ('fail', 'continue'):
            continue
        key = (row['data_identity']['split_group'], public_key(row), tuple(row['image_sha256'][:len(row['images'])-len(payload(row)['history'])]))
        buckets[key][label].append(row)
    pairs = []
    for bucket in buckets.values():
        for a, b in zip(bucket['fail'], bucket['continue']):
            pairs.append({'stop_id': a['id'], 'continue_id': b['id'], 'step': a['decision_index'],
                          'remaining': BUDGET-a['decision_index'],
                          'split_group': a['data_identity']['split_group'],
                          'match': 'same_object_references_context_actions_cameras_budget; actual_candidate_images_differ'})
    return pairs


def audit(root, baseline, processor_model=None):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete render')
    traces = read(root / 'trajectories.jsonl')
    expected = {}
    ambiguous = Counter()
    observed_gains = []
    order_pixels = {}
    groups = defaultdict(set)
    for t in traces:
        split, uid = t['split'], t['id'].split(':')[0]
        groups[split].add(t['data_identity']['split_group'])
        if t['data_identity']['formal_training'] is not True:
            raise ValueError('Non-formal source')
        order_id = int(t['id'].rsplit('order', 1)[1])
        motion = 'reference_states' in t['context']
        if motion:
            reference_states = t['context']['reference_states']
            state = reference_states[0]
            order = schedules(state['joint'], reference_states[-1]['detent'])[order_id]
            if [h['action'] for h in t['history']] != [action_json(a) for a in order]:
                raise ValueError('Motion order changed')
        elif [h['action']['view_id'] for h in t['history']] != list(ORDERS[order_id]):
            raise ValueError('Order or legal view changed')
        evidence, visited = [], set()
        for i, h in enumerate(t['history']):
            if h['step'] != i:
                raise ValueError('Illegal action or noncausal step')
            if motion:
                state = advance(state, h['action'])
                ref_index = reference_states.index(state)
                if h['reference_index'] != ref_index:
                    raise ValueError('Observed state/reference mismatch')
            else:
                ref_index = VIEWS.index(h['action']['view_id'])
            image = root / h['image_ref']
            digest = sha(image)
            key = (uid, ref_index)
            if key in order_pixels and order_pixels[key] != digest:
                raise ValueError('Different real replay pixels across orders')
            order_pixels[key] = digest
            ev = image_evidence((root / t['references'][ref_index]).read_bytes(), image.read_bytes())
            evidence.append(ev)
            observed_gains.append({'trajectory': t['id'], 'step': i, 'action': h['action'],
                                   'ex_post_evidence_gain': int(ev['state'] == 'sufficient' and
                                                              not any(e['state'] == 'sufficient' for e in evidence[:-1])),
                                   'new_reference_state_gain': int(ref_index not in visited),
                                   'use': 'audit_only_not_action_target'})
            visited.add(ref_index)
            row = make_row(uid, order_id, t['context'], t['references'], t['history'][:i+1],
                           evidence[:], t['data_identity'])
            if row is None:
                ambiguous[split] += 1
            else:
                row['image_sha256'] = [sha(root / p) for p in row['images']]
                expected[row['id']] = row
        if evidence != t['evidence']:
            raise ValueError('Saved evidence differs from public pixels')
    before_dedup = len(expected)
    raw_expected = list(expected.values())
    expected = {r['id']: r for r in deduplicate(expected.values())}
    canonical = {(r['messages'][0]['content'], tuple(r['image_sha256'])): r['id'] for r in expected.values()}
    aliases = {r['id']: canonical[(r['messages'][0]['content'], tuple(r['image_sha256']))] for r in raw_expected}
    (root / 'prefix-aliases.audit.jsonl').write_text(''.join(
        json.dumps({'trace_prefix_id': raw_id, 'training_row_id': canonical_id})+'\n'
        for raw_id, canonical_id in aliases.items()))
    if groups['train'] & groups['validation']:
        raise ValueError('Object split leakage')
    result = {'status': 'complete', 'traces': len(traces), 'real_observations': len(observed_gains),
              'exact_public_duplicates_removed': before_dedup-len(expected),
              'ambiguous_prefixes_excluded': dict(ambiguous), 'split_overlap': 0,
              'label_recomputation': 'all rows byte-structure equal from public images and causal prefixes',
              'order_replay_check': 'same candidate/reference-state has identical PNG SHA across independent executions',
              'scope': manifest['scope'], 'splits': {}}
    seen = set()
    all_rows = {}
    processor_checks = {}
    if processor_model:
        import torch
        from transformers import AutoProcessor
        torch.set_num_threads(2)
        processor = AutoProcessor.from_pretrained(processor_model, local_files_only=True, trust_remote_code=False)
    for split in ('train', 'validation'):
        rows = []
        # Reuse production loader: PNG closure, SHA, identity and completeness.
        lengths, targets = [], []
        for row, images in load_examples(root, split, 'stopping', require_formal_training=True):
            if row['id'] in seen or row != expected[row['id']]:
                raise ValueError('Duplicate or contaminated target/prefix')
            if row['data_identity']['split_group'] not in groups[split]:
                raise ValueError('Wrong split')
            seen.add(row['id'])
            rows.append(row)
            if processor_model:
                from turnitover.oracle.training import tokenize_example
                batch = tokenize_example(processor, row, images)
                lengths.append(int(batch['input_ids'].shape[1]))
                targets.append(int((batch['labels'] != -100).sum()))
        all_rows[split] = rows
        if processor_model:
            processor_checks[split] = {'rows': len(lengths), 'tokens_min_max': [min(lengths), max(lengths)],
                                       'assistant_tokens_min_max': [min(targets), max(targets)],
                                       'prefix_mask_and_nonempty_target': 'passed'}
        pairs = pair_rows(rows)
        within_sample = defaultdict(set)
        order_groups = defaultdict(dict)
        for r in raw_expected:
            if r['data_identity']['split_group'] in groups[split] and payload(r)['remaining'] > 0:
                within_sample[(r['source_sample_id'], r['decision_index'])].add(r['stop_target'])
                order_groups[(r['source_sample_id'], r['decision_index'])][r['stop_target']] = r
        order_pairs = [{'stop_id': aliases[g['stop']['id']], 'continue_id': aliases[g['continue']['id']],
                        'stop_trace_prefix_id': g['stop']['id'], 'continue_trace_prefix_id': g['continue']['id'],
                        'source_sample_id': uid, 'step': step, 'remaining': BUDGET-step,
                        'split_group': g['stop']['data_identity']['split_group'],
                        'match': 'same_candidate_and_budget_different_real_action_order'}
                       for (uid, step), g in order_groups.items() if 'stop' in g and 'continue' in g]
        (root / f'{split}.order-pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in order_pairs))
        first_evidence = Counter()
        for t in traces:
            if t['split'] == split:
                first_evidence[str(next((i+1 for i, e in enumerate(t['evidence'])
                                         if e['state'] == 'sufficient'), None))] += 1
        (root / f'{split}.pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
        selected = {p[k] for p in pairs for k in ('stop_id', 'continue_id')}
        (root / f'{split}.matched.stopping.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['id'] in selected))
        first_stop_rows = [r for r in rows if not r['evidence_steps'] or
                           min(r['evidence_steps']) == r['decision_index']-1]
        (root / f'{split}.first-stop.stopping.jsonl').write_text(
            ''.join(json.dumps(r)+'\n' for r in first_stop_rows))
        old = read(baseline / f'{split}.stopping.jsonl')
        result['splits'][split] = {'v5': stats(rows), 'v4': stats(old),
            'v4_matched_pairs': len(pair_rows(old)),
            'v4_matched_steps': dict(Counter(p['step'] for p in pair_rows(old))),
            'first_stop_rows': len(first_stop_rows),
            'first_stop_matched_pairs': len(pair_rows(first_stop_rows)),
            'first_stop_matched_steps': dict(Counter(p['step'] for p in pair_rows(first_stop_rows))),
            'matched_pairs': len(pairs), 'matched_rows': len(selected),
            'matched_objects': len({p['split_group'] for p in pairs}),
            'matched_steps': dict(Counter(p['step'] for p in pairs)),
            'matched_budget_only_accuracy_ceiling': .5 if pairs else None,
            'matched_all_nonimage_features_accuracy_ceiling': .5 if pairs else None,
            'termination_kinds': dict(Counter(str(r['termination_kind']) for r in rows)),
            'first_evidence_step_per_trace': dict(first_evidence),
            'same_candidate_same_step_opposite_order_label_groups': len(order_pairs),
            'same_candidate_order_pair_objects': len({p['split_group'] for p in order_pairs}),
            'post_sufficient_collection_prefixes': sum(bool(r['evidence_steps']) and
                min(r['evidence_steps']) < r['decision_index']-1 for r in rows)}
    if seen != set(expected):
        raise ValueError('Missing causal records')
    for name, train, val in (
            ('v5', all_rows['train'], all_rows['validation']),
            ('v4', read(baseline / 'train.stopping.jsonl'), read(baseline / 'validation.stopping.jsonl'))):
        lookup = defaultdict(Counter)
        for r in train:
            lookup[stage(r)][(r['stop_target'], r['stop_outcome'])] += 1
        predictions = {k: v.most_common(1)[0][0] for k, v in lookup.items()}
        correct = sum(predictions.get(stage(r)) == (r['stop_target'], r['stop_outcome']) for r in val)
        result.setdefault('train_fitted_budget_baseline', {})[name] = {'correct': correct, 'total': len(val),
                                                                   'accuracy': correct/len(val)}
    keys = lambda rows: {(r['messages'][0]['content'], tuple(r['image_sha256'])) for r in rows}
    result['public_input_split_overlap'] = len(keys(all_rows['train']) & keys(all_rows['validation']))
    if result['public_input_split_overlap']:
        raise ValueError('Public inputs overlap across splits')
    result['audit_source_sha256'] = sha(Path(__file__))
    result['decision_source_sha256'] = sha(Path('turnitover/oracle/visual_decisions.py'))
    result['processor_checks'] = processor_checks or 'not_requested'
    (root / 'action-outcomes.audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in observed_gains))
    result['hashes'] = {p.name: sha(p) for p in root.glob('*.jsonl')}
    result['baseline_manifest_sha256'] = sha(baseline / 'manifest.json')
    result['training_ready'] = 'bounded public-reference stopping pilot; no pass, full joint coverage or optimal action labels'
    (root / 'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--processor-model', type=Path)
    args = p.parse_args()
    audit(args.dataset, args.baseline, args.processor_model)
