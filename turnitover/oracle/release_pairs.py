"""Normalize release frames to one visible view and three matched state pairs."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import zipfile

from turnitover.oracle.evidence import pixel_signal, pixels


def _exposure_table(case, members, frames, labels, audits):
    """Single-defect difference is a proxy; composites require label-specific audit."""
    count = len(labels)
    signals = [pixel_signal(pixels(frames[i]), pixels(frames[i + 1])) for i in (0, 2, 4)]
    if count <= 1:
        return [[0] if count and signal['exposed'] else [] for signal in signals], signals
    audit = audits.get(case.gold['case_id'], {})
    expected = {'reference_sha': case.gold['reference_sha'],
                'candidate_sha': case.gold['candidate_sha'],
                'image_sha256': [hashlib.sha256(data).hexdigest() for data in frames],
                'labels_sha256': hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()}
    view = members[0].view
    audit = audit.get('views', {}).get(view, {})
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{case.gold['case_id']}: composite needs hash-bound per-label exposure audit for {view}")
    table = audit.get('pair_label_indices')
    if (not isinstance(table, list) or len(table) != 3
            or any(not isinstance(row, list) or len(row) != len(set(row))
                   or any(type(index) is not int or not 0 <= index < count for index in row)
                   for row in table)):
        raise ValueError('Invalid per-label paired exposure table')
    if not audit.get('method') or not audit.get('provenance'):
        raise ValueError('Composite exposure audit needs method and provenance')
    if any(row and not signal['exposed'] for row, signal in zip(table, signals, strict=True)):
        raise ValueError('Audited exposure contradicts matched reference/candidate pixels')
    return table, signals


def prepare_pairs(case, members, labels, *, easy_board_roots=(), exposure_audits=None):
    """Read genuine frames; never fabricate states or infer composite label coverage."""
    with zipfile.ZipFile(case.archive) as source:
        originals = {(m.view, m.kind, m.state): (m, source.read(m.path)) for m in members}
    views = list(dict.fromkeys(m.view for m in members))
    provenance = {}
    if case.level == 'easy':
        from turnitover.oracle.easy_pair_frames import easy_mid_frames
        view = views[0]
        old = {f'{kind}_{state}': data for (_, kind, state), (_, data) in originals.items()}
        mids, provenance = easy_mid_frames(case.gold['case_id'], old, list(easy_board_roots))
        for kind in ('reference', 'candidate'):
            base = originals[(view, kind, 'rest')][0]
            originals[(view, kind, 'mid')] = (replace(base, role=f'{kind}_mid', state='mid'), mids[f'{kind}_mid'])
    candidates, errors = [], []
    for view in views:
        selected = [originals[(view, kind, state)] for state in ('rest', 'mid', 'limit')
                    for kind in ('reference', 'candidate')]
        selected_members, frames = map(list, zip(*selected, strict=True))
        try:
            table, signals = _exposure_table(case, selected_members, frames, labels, exposure_audits or {})
        except ValueError as exc:
            errors.append(str(exc))
            continue
        covered = {index for row in table for index in row}
        if labels and covered != set(range(len(labels))):
            errors.append(f'{view}: not all annotated defects are exposed in one view')
            continue
        candidates.append((sum(s['image_l1'] for s in signals), selected_members, frames, table, signals))
    if not candidates:
        raise ValueError(f"{case.gold['case_id']}: no qualified single-view paired input; {'; '.join(errors)}")
    _, selected_members, frames, table, signals = max(candidates, key=lambda row: row[0])
    audit = {'selected_view': selected_members[0].view, 'pair_signals': signals,
             'exposure_method': 'matched_pair_pixel_difference_single_defect' if len(labels) <= 1
             else 'supplied_per_label_audit', 'easy_mid': provenance}
    if len(labels) > 1:
        source = exposure_audits[case.gold['case_id']]['views'][selected_members[0].view]
        audit['per_label_audit'] = source
        audit['per_label_audit_sha256'] = hashlib.sha256(
            json.dumps(source, sort_keys=True).encode()).hexdigest()
    return selected_members, frames, table, audit


def copy_pairs(case, output: Path, members, frames):
    paths, digests = [], []
    for index, (member, data) in enumerate(zip(members, frames, strict=True)):
        state = 'zero' if member.state == 'rest' else member.state
        relative = f"images/{case.gold['case_id']}/{index:02d}-{member.kind}_{state}.png"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        paths.append(relative)
        digests.append(hashlib.sha256(data).hexdigest())
    history = [{'step': index, 'state': state,
                'action': {'type': 'request_view', 'view_id': members[0].view},
                'reference_image_ref': paths[2 * index], 'image_ref': paths[2 * index + 1]}
               for index, state in enumerate(('zero', 'mid', 'limit'))]
    return history, digests
