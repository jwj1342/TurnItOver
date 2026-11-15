"""Independent repair metrics; recognition F1 and public PASS are not scores."""
from __future__ import annotations

import math
import hashlib

from turnitover.taxonomy import load_taxonomy


def classify_exposure(index, evidence, history):
    """Classify a label INSTANCE from frozen evidence, never from its type name."""
    if evidence.get('frozen') is not True:
        return 'unknown'
    mapping = evidence.get('label_evidence_steps', {})
    steps = mapping.get(str(index))
    if not isinstance(steps, list) or not steps:
        return 'unknown'
    by_step = {row['step']: row for row in history}
    if any(step not in by_step for step in steps):
        return 'unknown'
    exposed = [by_step[step] for step in steps]
    visual = [row for row in exposed if row.get('action', {}).get('type') == 'request_view']
    if any(row.get('state') in {'zero', 'rest'} for row in visual):
        return 'visible_at_rest'
    if any(row.get('payload', {}).get('measurement_kind') in {'public_tick_trace', 'temporal'}
           or row.get('payload', {}).get('kind') == 'public_tick_trace'
           or row.get('time') is not None for row in exposed):
        return 'temporal'
    if visual:
        # No rest example or no known state is insufficient to establish motion necessity.
        rest_views = {row['action'].get('view_id') for row in history
                      if row.get('action', {}).get('type') == 'request_view'
                      and row.get('state') in {'zero', 'rest'}}
        if (all(row.get('state') in {'mid', 'midpoint', 'limit', 'endpoint'} for row in visual)
                and all(row['action'].get('view_id') in rest_views
                        and row['action'].get('view_id') is not None for row in visual)):
            return 'needs_motion'
        return 'unknown'
    if all(row.get('action', {}).get('type') == 'query_runtime' for row in exposed):
        return 'probe'
    return 'unknown'


def score_audit(labels, measurements, program_checks, required_checks, *,
                program_sha256, exposure=None):
    """Join fresh native checker results by label index, preserving multi-defects.

    measurements: index -> {passed: bool|None, checker_id, ...}. required_checks
    comes from a frozen evaluator contract and includes regression checks beyond
    injected defects. Missing evidence stays unknown. A known failure stays false
    even when another checker is missing; complete coverage is reported separately.
    """
    ids = set(load_taxonomy().ids)
    if any(label.get('defect_id') not in ids for label in labels):
        raise ValueError('Unknown private defect label')
    if not required_checks or len(set(required_checks)) != len(required_checks):
        raise ValueError('Frozen nonempty independent checker roster required')
    defects = []
    for index, label in enumerate(labels):
        item = measurements.get(index, {})
        passed = item.get('passed')
        if type(passed) not in (bool, type(None)):
            raise ValueError('Per-defect pass must be boolean or null')
        if passed is not None and not item.get('checker_id'):
            raise ValueError('Scored defect requires checker provenance')
        defects.append({'label_index': index, 'defect_id': label['defect_id'],
                        'parts': label.get('parts', []), 'resolved': passed,
                        'checker_id': item.get('checker_id'),
                        'exposure': (exposure or {}).get(index, 'unknown')})
    outcomes = [row['resolved'] for row in defects]
    missing = []
    for name in required_checks:
        value = program_checks.get(name)
        if type(value) not in (bool, type(None)):
            raise ValueError('Independent program check must be boolean or null')
        outcomes.append(value)
        if value is None:
            missing.append(name)
    complete = all(value is not None for value in outcomes)
    passed = False if False in outcomes else True if complete else None
    return {'passed': passed, 'coverage_complete': complete,
            'program_sha256': program_sha256, 'defects': defects,
            'program_checks': program_checks, 'missing_checks': missing,
            'missing_defects': [row['label_index'] for row in defects if row['resolved'] is None]}


def checker_coverage(supported):
    """Inventory all 29 labels; a callback name alone is not coverage evidence."""
    ids = set(load_taxonomy().ids)
    if set(supported) - ids:
        raise ValueError('Unknown supported defect type')
    return {'supported': sorted(supported), 'missing': sorted(ids - set(supported)),
            'complete': ids == set(supported)}


def runtime_budget_predicate(label, evidence, *, max_triangles, max_draw_calls):
    """Reuse RuntimeBudgetChecker with frozen PUBLIC caps over all three states.

    This covers only the two budget labels, not lifecycle, temporal or geometry.
    No cap or tolerance is fitted on repaired validation programs.
    """
    from turnitover.checkers.base import Evidence
    from turnitover.checkers.runtime_budget import RuntimeBudgetChecker
    names = {'runtime.triangle_budget': 'triangles', 'runtime.draw_calls': 'draw_calls'}
    if label['defect_id'] not in names:
        raise ValueError('Not a runtime budget defect')
    caps = {'runtime.triangle_budget': max_triangles, 'runtime.draw_calls': max_draw_calls}
    if (type(caps[label['defect_id']]) is not int or caps[label['defect_id']] < 0
            or any(value is not None and (type(value) is not int or value < 0)
                   for value in caps.values())):
        raise ValueError('Frozen public runtime caps are required')
    # Only the selected invariant is scored. A missing companion cap is not a
    # fabricated public requirement; its unused checker threshold is immaterial.
    checker = RuntimeBudgetChecker(max_triangles or 0, max_draw_calls or 0)
    measurements = [row for row in evidence.get('runtime', [])
                    if row.get('role') == 'candidate' and row.get('kind') == 'public_stats']
    result = {'passed': None, 'checker_id': checker.checker_id,
              'caps': {'max_triangles': max_triangles, 'max_draw_calls': max_draw_calls}}
    if [row.get('state') for row in measurements] != ['zero', 'mid', 'limit']:
        return {**result, 'reason': 'Incomplete candidate state measurements'}
    passes = []
    for row in measurements:
        stats = row.get('value')
        if not isinstance(stats, dict):
            return {**result, 'reason': 'Missing runtime stats'}
        checked = checker.check(Evidence(evidence['program_sha256']['candidate'], True, None,
                                         {}, stats, {}))
        if checked.score is None:
            return {**result, 'reason': 'Incomplete runtime evidence'}
        passes.append(next(inv.passed for inv in checked.invariants if inv.name == names[label['defect_id']]))
    return {**result, 'passed': all(passes)}


def make_repair_audit(replay, labels, defect_checkers, program_checkers, required_checks,
                      *, exposure=None):
    """Bind existing native predicates to fresh source-bound evidence.

    defect_checkers maps taxonomy ID to (label, replay_evidence)->measurement;
    program_checkers maps frozen check ID to replay_evidence->bool|None.
    Missing predicates are explicit gaps, not permissive stub implementations.
    Neither checker results nor labels are part of the public judge contract.
    """
    def audit(source):
        evidence = replay(source)
        sha = hashlib.sha256(source.encode()).hexdigest()
        if evidence.get('status') != 'complete' or evidence.get('program_sha256', {}).get('candidate') != sha:
            raise ValueError('Independent scoring requires complete current-source replay')
        gate = evidence.get('gate') or {}
        rejected = gate.get('passed') is False
        # A deterministic compile/render rejection is a known unsuccessful
        # program, not a checker crash. Do not attempt unavailable mesh probes.
        measurements = {} if rejected else {
            index: defect_checkers[label['defect_id']](label, evidence)
            for index, label in enumerate(labels) if label['defect_id'] in defect_checkers}
        checks = {name: (False if name == 'compile_render' else None) if rejected else checker(evidence)
                  for name, checker in program_checkers.items()}
        result = score_audit(labels, measurements, checks, required_checks,
                             program_sha256=sha, exposure=exposure)
        result['checker_coverage'] = checker_coverage(defect_checkers)
        result['gate_passed'] = False if rejected else gate.get('passed')
        if rejected:
            result['passed'] = False
        result['exposure_complete'] = all(row['exposure'] in {
            'needs_motion', 'visible_at_rest', 'probe', 'temporal'} for row in result['defects'])
        result['table1_eligible'] = (result['coverage_complete']
            and result['checker_coverage']['complete'] and result['exposure_complete'])
        return result
    return audit


def summarize_condition(rows):
    """No survivor filtering: failure/unknown counts accompany every denominator."""
    total = len(rows)
    passed = [r.get('final_passed') if r.get('status') == 'complete' else None for r in rows]
    known = sum(type(value) is bool for value in passed)
    success = sum(value is True for value in passed)
    resolved = {name: {'total': 0, 'known': 0, 'resolved': 0} for name in
                ('needs_motion', 'visible_at_rest', 'probe', 'temporal', 'unknown')}
    by_type = {}
    roster_complete = True
    for row in rows:
        # audit_roster must be supplied for incomplete runs to retain defect denominators.
        defects = row.get('final_audit', {}).get('defects', row.get('audit_roster', []))
        if not defects:
            roster_complete = False
        for defect in defects:
            group = defect.get('exposure', 'unknown')
            if group not in resolved:
                raise ValueError('Unknown exposure class')
            counters = (resolved[group], by_type.setdefault(defect['defect_id'],
                         {'total': 0, 'known': 0, 'resolved': 0}))
            value = defect.get('resolved') if row.get('status') == 'complete' else None
            for counter in counters:
                counter['total'] += 1
                counter['known'] += type(value) is bool
                counter['resolved'] += value is True
    for counter in [*resolved.values(), *by_type.values()]:
        counter['rate'] = counter['resolved'] / counter['total'] if roster_complete and counter['total'] and counter['known'] == counter['total'] else None
    successes = [row for row, value in zip(rows, passed) if value is True]
    def average(field):
        values = [row.get(field) for row in successes]
        if not values or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            return None
        return sum(values) / len(values)
    return {'target': total, 'defect_roster_complete': roster_complete,
            'known_success_outcomes': known, 'successes': success,
            'failed_or_interrupted': sum(r.get('status') != 'complete' for r in rows),
            'success_rate': success / total if total and known == total else None,
            'success_lower_bound': success / total if total else None,
            'resolved': resolved, 'resolved_by_type': by_type,
            'cost_to_success': {'runs': len(successes),
                'iterations': average('first_success_round'), 'tokens': average('tokens_to_first_success')}}


def summarize_table1(rows, generators, *, expected_conditions=None):
    """Require an explicitly paired generator/case/repetition matrix before averaging."""
    if len(generators) != 2 or len(set(generators)) != 2:
        raise ValueError('Table 1 requires two distinct frozen generators')
    from turnitover.repair.table1_judges import CONDITIONS
    expected = set(CONDITIONS if expected_conditions is None else expected_conditions)
    conditions = dict.fromkeys(row['condition'] for row in rows)
    if set(conditions) != expected or any(row['generator'] not in generators for row in rows):
        raise ValueError('Table 1 condition/generator matrix is incomplete or unexpected')
    protocols = {row.get('comparison_protocol_sha256') for row in rows}
    if len(protocols) != 1 or None in protocols:
        raise ValueError('Table 1 requires one frozen comparison protocol across all runs')
    for generator in generators:
        configs = [row.get('generator_config') for row in rows if row['generator'] == generator]
        if not configs or configs[0] is None or any(config != configs[0] for config in configs):
            raise ValueError('Each generator must keep the same frozen config across all judges')
    result = {}
    common_roster = None
    for condition in conditions:
        configs = [row.get('judge_config') for row in rows if row['condition'] == condition]
        if any(config != configs[0] for config in configs):
            raise ValueError('Each judge must use the same frozen config with both generators')
        groups, rosters = {}, []
        for generator in generators:
            selected = [row for row in rows if row['condition'] == condition and row['generator'] == generator]
            roster = [(r['id'], r['repetition']) for r in selected]
            if len(roster) != len(set(roster)):
                raise ValueError('Duplicate case/repetition in Table 1')
            rosters.append(set(roster))
            groups[generator] = summarize_condition(selected)
        if not rosters[0] or rosters[0] != rosters[1]:
            raise ValueError('Generators must use identical case/repetition rosters')
        if common_roster is not None and common_roster != rosters[0]:
            raise ValueError('All judges must use identical case/repetition rosters')
        common_roster = rosters[0]
        result[condition] = {'generators': groups}
        result[condition]['table1_eligible'] = all(
            row.get('final_audit', {}).get('table1_eligible') is True
            for row in rows if row['condition'] == condition)
        def mean(values):
            return sum(values) / 2 if all(value is not None for value in values) else None
        result[condition]['resolved'] = {group: mean([groups[g]['resolved'][group]['rate'] for g in generators])
                                         for group in ('needs_motion', 'visible_at_rest')}
        result[condition]['cost_to_success'] = {field: mean([groups[g]['cost_to_success'][field] for g in generators])
                                                for field in ('iterations', 'tokens')}
        if condition == 'none':
            result[condition]['cost_to_success']['iterations'] = None
    return result
