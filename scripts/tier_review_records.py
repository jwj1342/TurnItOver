"""Validate and classify actual blind-review rows without inventing approvals."""
from __future__ import annotations

from datetime import datetime


def classify(raw: dict, metadata: dict) -> tuple[str, dict]:
    if not isinstance(raw.get('notes'), str):
        raise ValueError('Review notes must be a string')
    reviewer = raw.get('reviewer')
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 128:
        raise ValueError('A real return must contain a reviewer identifier')
    decision = raw.get('decision')
    reviewed_at = raw.get('reviewed_at')
    if decision is None:
        if reviewed_at is not None:
            raise ValueError('Pending judgment must not claim a review date')
        return 'pending', {'status': 'pending', 'reviewer': reviewer, 'reviewed_at': None,
                           'verdict': None, 'difficulty': None}
    if decision not in ('easy', 'medium', 'hard', 'clean', 'invalid'):
        raise ValueError('Unknown review decision')
    try:
        datetime.fromisoformat(reviewed_at.replace('Z', '+00:00'))
    except (AttributeError, TypeError, ValueError):
        raise ValueError('Completed review needs an ISO date or timestamp') from None
    if not raw['notes'].strip():
        raise ValueError('Completed review requires observation notes')
    expected = 'clean' if metadata['kind'] == 'clean' else metadata['tier']
    status = 'approved' if decision == expected else 'rejected'
    normalized = {'status': 'passed' if status == 'approved' else 'rejected',
                  'reviewer': reviewer, 'reviewed_at': reviewed_at,
                  'verdict': 'clean' if decision == 'clean' else 'defect' if decision != 'invalid' else 'invalid',
                  'difficulty': decision}
    return status, normalized


def verify_sets(raw_rows: list, mapping_rows: list, cases: list, package_sha: str) -> None:
    for rows, name, key in ((raw_rows, 'return', 'case_id'), (mapping_rows, 'mapping', 'case_id'),
                            (cases, 'package', 'case_id')):
        if not rows or len({row[key] for row in rows}) != len(rows):
            raise ValueError(f'Empty or duplicate opaque IDs in {name}')
    expected = {row['case_id'] for row in cases}
    if {row['case_id'] for row in raw_rows} != expected or {row['case_id'] for row in mapping_rows} != expected:
        raise ValueError('Return/mapping/package opaque ID sets differ')
    if any(row.get('review_package_sha256') != package_sha for row in raw_rows):
        raise ValueError('Raw review row is bound to a different package')
    by_id = {row['case_id']: row for row in cases}
    for row in mapping_rows:
        case = by_id[row['case_id']]
        shown = [item['path'] for group in ('primary', 'supplementary') for item in case[group]]
        mapped = [item['path'] for item in row['media']]
        if sorted(shown) != sorted(mapped) or len(set(mapped)) != len(mapped):
            raise ValueError('Custody media differs from the opaque case shown to reviewer')
