"""Offline grounding analysis on the three-tier-900 validation split (no model calls).

(a) Unexposed finding rate per model/supervision/seed: 1 - (#predicted findings whose defect_id is a gold label of
    the case AND that label is exposed by at least one shown cell) / (#predicted findings). Every finding on a clean
    case is unexposed. Exposure per label comes from the evidence-constrained target of the case's validation episode
    (findings' evidence_steps -> image refs -> cells); under fixed full observation every cell is shown.
(b) Share of cases with at least one finding, and the share of clean cases with a finding.
(c) Evidence support recomputed from predictions (supported_findings / predicted findings) against her metrics.json.
(d) Exposure ceilings by action type for the 60 defective validation cases from the inventory and exposure records:
    viewpoints only (rest-state cells across views), joints only (cells at the initial view across states), both (all
    cells), plus Random and Fixed at the inventory budget (her policy code), per-case label fraction averaged over cases.
Usage: python analysis/unexposed_rate.py  (writes out/RESULTS-grounding-offline.md)
"""
from __future__ import annotations
import os
import json, sys
from collections import defaultdict
from pathlib import Path
TIO = Path(os.environ.get('TIO_RUNS', '/scratch/jwj/tio-runs'))
EXPORT = TIO / 'data900/three-tier-900-sft-v1/episode_export'
INV = TIO / 'action900/inventory/case-inventory.private.jsonl'
sys.path.insert(0, str(TIO / 'action900/scripts'))
from evaluate_900_oracle_baselines import policy_cells  # noqa: E402  (her fixed/random policies)
TIERS = ('easy', 'medium', 'hard', 'all')


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def payload_of(ep: dict) -> dict:
    c = ep['messages'][0]['content']
    return json.loads(c[c.index('{"context"'):])


def exposing_images(ep: dict) -> dict[str, set[str]]:
    """defect_id -> images cited by the evidence-constrained target."""
    p = payload_of(ep)
    step_img = {h['step']: h['image_ref'] for h in p['history']}
    target = json.loads(ep['messages'][-1]['content'])['verdict']
    out: dict[str, set[str]] = defaultdict(set)
    for f in target['findings']:
        for s in f['evidence_steps']:
            out[f['defect_id']].add(step_img[s])
    return out


def pct(values):
    values = [v for v in values if v is not None]
    return 100 * sum(values) / len(values) if values else None


def fmt(v):
    return '–' if v is None else f'{v:.1f}'


def fmt_seeds(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return '–'
    if len(vals) == 1:
        return f'{vals[0]:.1f}'
    return f'{sum(vals)/len(vals):.1f} [{min(vals):.1f}, {max(vals):.1f}]'


def discover():
    runs = []
    for m in sorted(TIO.glob('out/level900-sft-*/*/seed*/*-diagnosis/eval-validation/predictions.jsonl')):
        runs.append((m.parts[-5], m.parts[-3].split('-')[0], m.parts[-4][4:], m))
    for m in sorted(TIO.glob('out/level900-base-*/*/base-validation/predictions.jsonl')):
        runs.append((m.parts[-3], 'zero-shot', '-', m))
    return runs


def main() -> None:
    episodes = {e['episode_id']: e for e in rows(EXPORT / 'validation.episodes.jsonl')}
    ann = {a['id']: a for a in rows(EXPORT / 'annotations.private.jsonl') if a['split'] == 'validation'}
    gold = {cid: {lab['defect_id'] for lab in a['labels']} for cid, a in ann.items()}
    exposed = {cid: {d for d, imgs in exposing_images(e).items() if imgs} for cid, e in episodes.items()}
    tier = {cid: e['difficulty'] for cid, e in episodes.items()}
    runs = discover()
    per = {}   # (size, sup, seed) -> {tier: metrics}
    checked = False
    for size, sup, seed, path in runs:
        preds = rows(path)
        acc = defaultdict(lambda: defaultdict(list))
        tot = defaultdict(int); unexp = defaultdict(int); supp = defaultdict(int); nocase_f = defaultdict(int)
        for r in preds:
            cid = r['episode_id']
            findings = r.get('predicted_findings') or []
            ok_labels = gold[cid] & exposed[cid]
            n_unexp = sum(f[0] not in ok_labels for f in findings)
            for t in (tier[cid], 'all'):
                tot[t] += len(findings); unexp[t] += n_unexp; supp[t] += r.get('supported_findings', 0) or 0
                nocase_f[t] += len(findings) if not ok_labels else 0
                acc[t]['any'].append(bool(findings))
                if gold[cid]:
                    acc[t]['any_defect'].append(bool(findings))
                    acc[t]['unexp_defect_n'].append(n_unexp); acc[t]['defect_f_n'].append(len(findings))
                else:
                    acc[t]['clean_any'].append(bool(findings))
            if not checked and sup == 'evidence' and findings and size == '2B':
                print(f'[hand check] {size} {sup} seed {seed} case {cid} ({tier[cid]}): gold={sorted(gold[cid])} exposed={sorted(exposed[cid])} '
                      f'predicted={[f[0] for f in findings]} -> unexposed {n_unexp}/{len(findings)}; her supported_findings={r.get("supported_findings")}')
                checked = True
        m = json.loads((path.parent / 'metrics.json').read_text())
        out = {}
        for t in TIERS:
            a = acc[t]
            out[t] = dict(
                unexposed=(100 * unexp[t] / tot[t]) if tot[t] else None,
                unexposed_defect=(100 * sum(a['unexp_defect_n']) / sum(a['defect_f_n'])) if sum(a['defect_f_n']) else None,
                findings=tot[t], any_case=pct(a['any']), clean_any=pct(a['clean_any']),
                support=(100 * supp[t] / tot[t]) if tot[t] else None,
                nocase=(100 * nocase_f[t] / tot[t]) if tot[t] else None,
                her_support=(lambda v: None if v is None else 100 * v)((m['evidence_support_rate'] if t == 'all' else m['by_difficulty'][t]['evidence_support_rate']) if m.get('status') == 'complete' else None))
        per[(size, sup, seed)] = out
    # ---- markdown
    lines = ['# Grounding analysis, offline (validation 120 cases, fixed full observation)', '',
             f'Runs discovered: {len(runs)}. Generated by analysis/unexposed_rate.py.', '',
             '## A. Unexposed finding rate (%), mean [min, max] over seeds', '',
             '| Model | Supervision | ' + ' | '.join(f'{t} unexp. | {t} unexp. (defect cases) | {t} on cases w/o exposed label' for t in TIERS) + ' |',
             '|---|---|' + '---:|' * (3 * len(TIERS))]
    groups = defaultdict(list)
    for (size, sup, seed), out in per.items():
        groups[(size, sup)].append(out)
    order = sorted(groups, key=lambda k: (k[0], {'zero-shot': 0, 'ordinary': 1, 'evidence': 2}[k[1]]))
    for key in order:
        cells = []
        for t in TIERS:
            cells.append(fmt_seeds([o[t]['unexposed'] for o in groups[key]]))
            cells.append(fmt_seeds([o[t]['unexposed_defect'] for o in groups[key]]))
            cells.append(fmt_seeds([o[t]['nocase'] for o in groups[key]]))
        lines.append(f'| Qwen3-VL-{key[0]} | {key[1]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## B. Cases with any finding (%), clean cases with a finding (%), findings per run, evidence support (%) recomputed vs. hers (all tiers)', '',
              '| Model | Supervision | seed | any finding | clean with finding | #findings | support (recomputed) | support (her metrics) | ' + ' | '.join(f'any finding {t}' for t in TIERS[:3]) + ' |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (size, sup, seed) in sorted(per, key=lambda k: (k[0], {'zero-shot': 0, 'ordinary': 1, 'evidence': 2}[k[1]], k[2])):
        o = per[(size, sup, seed)]
        lines.append(f'| Qwen3-VL-{size} | {sup} | {seed} | {fmt(o["all"]["any_case"])} | {fmt(o["all"]["clean_any"])} | {o["all"]["findings"]} | '
                     f'{fmt(o["all"]["support"])} | {fmt(o["all"]["her_support"])} | ' + ' | '.join(fmt(o[t]['any_case']) for t in TIERS[:3]) + ' |')
    # ---- exposure ceilings by action type (no model)
    inv = [r for r in rows(INV) if r['split'] == 'validation' and r['supervision_eligible']]
    acc2 = defaultdict(lambda: defaultdict(list))
    for r in inv:
        cid = r['id']; e = episodes[cid]
        show = exposing_images(e)
        cell_of = {c['candidate_image']: (c['view_id'], c['state']) for c in r['observation_cells']}
        label_cells = {lab: {cell_of[i] for i in show.get(lab, ()) if i in cell_of} for lab in gold[cid]}
        init_view = r['initial_observation']['view_id']
        all_cells = {(c['view_id'], c['state']) for c in r['observation_cells']}
        sets = {
            'viewpoints only': {c for c in all_cells if c[1] == 'rest'},
            'joints only': {c for c in all_cells if c[0] == init_view},
            'viewpoints + joints (all cells)': all_cells,
            'Random (budget 1/1/2)': {(c['view_id'], c['state']) for c in policy_cells(r, 'random')[0]},
            'Fixed (budget 1/1/2)': {(c['view_id'], c['state']) for c in policy_cells(r, 'fixed')[0]},
        }
        for name, got in sets.items():
            v = sum(bool(label_cells[lab] & got) for lab in gold[cid]) / len(gold[cid])
            acc2[name][r['difficulty']].append(v); acc2[name]['all'].append(v)
    lines += ['', '## C. Exposure by action type, 60 defective validation cases (% of gold labels exposed, per-case mean)', '',
              '| Observation set | easy | medium | hard | all |', '|---|---:|---:|---:|---:|']
    for name in ('Random (budget 1/1/2)', 'Fixed (budget 1/1/2)', 'viewpoints only', 'joints only', 'viewpoints + joints (all cells)'):
        lines.append(f'| {name} | ' + ' | '.join(fmt(pct(acc2[name][t])) for t in TIERS) + ' |')
    lines += ['', '## Notes', '',
              '- Unexposed finding rate: a predicted finding counts as exposed only if its defect_id is a gold label of the case and the evidence-constrained target cites at least one shown cell for that label. Findings are her signature-deduplicated predicted_findings. All findings on clean cases are unexposed. Under fixed full observation every cell is shown, so a label is exposed iff the target covers it.',
              '- "unexp. (defect cases)" restricts the denominator to findings made on defective cases.',
              '- "on cases w/o exposed label": share of all findings that were made on cases where no gold label is exposed by any shown cell (clean cases, plus defective cases whose labels the pixel proxy never exposes). Unlike the unexposed rate this does not count mislabeled findings on exposed defects, so it isolates invented findings.',
              '- Evidence support recomputed = sum of her per-row supported_findings / number of predicted findings; it should equal her evidence_support_rate.',
              '- Exposure sets: "viewpoints only" keeps cells in the rest state across views (what request_view alone can reach), "joints only" keeps cells at the initial view across states (what actuate_joint alone can reach). Easy and Medium have one view, so viewpoints only equals the initial cell there. Random and Fixed use her policy code at the inventory budget rule max(1, floor(unseen cells / 2)) = 1/1/2.',
              '- Exposure labels are a pixel proxy; a label whose exposing set is empty cannot be exposed by any policy, which caps Hard.']
    (TIO / 'out/RESULTS-grounding-offline.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
