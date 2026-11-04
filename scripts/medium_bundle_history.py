"""Validate original visibility judgments without inferring full qualification."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import re

if __package__:
    from .easy_release_archive import sha256
    from .validate_easy_bundle import _read_jsonl
else:
    from easy_release_archive import sha256
    from validate_easy_bundle import _read_jsonl

RAW_KEYS = ('case_id', 'reviewer', 'decision', 'notes', 'reviewed_at')
HISTORY = 'private/human_review_history'


def history_files():
    files = set()
    for number in (1, 2):
        prefix = f'{HISTORY}/round-{number:04d}'
        files.update(f'{prefix}/{name}' for name in ('raw.json', 'import.json', 'source_manifest.json'))
        files.add(f'{prefix}/' + ('source_spot_check_10.json' if number == 1 else 'source_review_page.html'))
    return files


def validate_history(root, gold):
    ledger = _read_jsonl(root / 'private/human_spot_review.jsonl')
    dispositions = _read_jsonl(root / 'private/human_spot_dispositions.jsonl')
    if (len(ledger) != 16 or len(dispositions) != 16 or len({r['case_id'] for r in ledger}) != 16
            or Counter(r['decision'] for r in ledger) != {'medium': 11, 'clean': 2, 'too_obvious': 3}):
        raise ValueError('Medium visibility history must preserve all sixteen real judgments')
    active = {r['asset_id'] for r in gold.values()}
    offset = 0
    for number, count in ((1, 10), (2, 6)):
        folder = root / HISTORY / f'round-{number:04d}'
        meta = json.loads((folder / 'import.json').read_text())
        raw = json.loads((folder / 'raw.json').read_text())
        page = 'source_spot_check_10.json' if number == 1 else 'source_review_page.html'
        key = 'source_spot_sha256' if number == 1 else 'source_review_page_sha256'
        pins = {'raw.json': 'raw_sha256', 'source_manifest.json': 'source_manifest_sha256', page: key}
        if any(sha256(folder / name) != meta[field] for name, field in pins.items()):
            raise ValueError('Original human review provenance changed')
        selected = (json.loads((folder / page).read_text())['case_ids'] if number == 1 else
                    re.findall(r'<section\s+data-id="(ms-[a-f0-9]+)"', (folder / page).read_text()))
        if (len(raw) != count or len(selected) != count or len(set(selected)) != count
                or {r['case_id'] for r in raw} != set(selected)):
            raise ValueError('Human judgments do not exactly cover their historical review page')
        end = offset + count
        for filename, field in (('human_spot_review.jsonl', 'ledger_sha256'),
                                ('human_spot_dispositions.jsonl', 'dispositions_sha256')):
            prefix = b''.join((root / 'private' / filename).read_bytes().splitlines(keepends=True)[:end])
            if hashlib.sha256(prefix).hexdigest() != meta[field]:
                raise ValueError('Original human ledger byte prefix changed')
        block = ledger[offset:end]
        if [{key: r[key] for key in RAW_KEYS} for r in block] != raw:
            raise ValueError('Raw judgments and the bound visibility ledger disagree')
        for row, disposition in zip(block, dispositions[offset:end]):
            if (row.get('reviewer') != 'human' or row.get('reviewer_type') != 'human'
                    or row.get('scope') != 'spot_visibility_only' or not isinstance(row.get('notes'), str)
                    or row.get('source_manifest_sha256') != meta['source_manifest_sha256']
                    or row.get(key) != meta[key]):
                raise ValueError('Human judgment identity or source binding changed')
            timestamp = datetime.fromisoformat(row['reviewed_at'].replace('Z', '+00:00'))
            if timestamp.utcoffset() is None:
                raise ValueError('Human review timestamps require a timezone')
            expected = ('spot_medium_observed_full_diagnostics_pending' if row['decision'] == 'medium'
                        else 'excluded_by_human_spot_review')
            if (any(disposition[k] != row[k] for k in ('case_id', 'asset_id', 'notes'))
                    or disposition['human_decision'] != row['decision']
                    or disposition['disposition'] != expected or disposition.get('reviewer_type') != 'human'):
                raise ValueError('Human disposition differs from the original judgment')
            if row['decision'] == 'medium':
                current = gold.get(row['case_id'])
                if (current is None or current['kind'] == 'clean'
                        or any(row['source_case'][k] != current[k]
                               for k in ('asset_id', 'candidate_sha', 'reference_sha'))
                        or row['source_case']['source_identity'] != current['source']['source_identity']):
                    raise ValueError('Approved visibility case lost its original candidate')
            elif row['asset_id'] in active or row['case_id'] in gold:
                raise ValueError('Human-rejected object returned to the candidate collection')
        offset = end
    return {'reviewed': 16, 'medium': 11, 'clean': 2, 'too_obvious': 3,
            'active_pending_visibility': 64, 'full_diagnostic_complete': 0}
