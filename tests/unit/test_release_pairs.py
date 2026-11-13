import io
import zipfile
from pathlib import Path

from PIL import Image
import pytest

from turnitover.oracle.release_sft_export import ImageMember, ReleaseCase
from turnitover.oracle.release_pairs import prepare_pairs


def _case(tmp_path, *, composite=False, visible=True):
    archive = tmp_path / 'frames.zip'
    members = []
    with zipfile.ZipFile(archive, 'w') as z:
        for view in ('front', 'side'):
            for kind in ('reference', 'candidate'):
                for state in ('rest', 'mid', 'limit'):
                    path = f'{view}/{kind}_{state}.png'
                    color = 'blue' if visible and view == 'side' and kind == 'candidate' else 'red'
                    stream = io.BytesIO()
                    Image.new('RGB', (4, 4), color).save(stream, format='PNG')
                    z.writestr(path, stream.getvalue())
                    members.append(ImageMember(f'{view}_{kind}_{state}', path, view, kind, state))
    case = ReleaseCase('hard', archive, 'hard', {
        'case_id': 'hard-test', 'reference_sha': 'a'*16, 'candidate_sha': 'b'*16}, {})
    labels = [{'defect_id': 'kinematics.joint_axis'}] * (2 if composite else 1)
    return case, members, labels


def test_hard_selects_visible_supplementary_view(tmp_path):
    case, members, labels = _case(tmp_path)
    selected, frames, table, audit = prepare_pairs(case, members, labels)
    assert len(frames) == len(selected) == 6
    assert {m.view for m in selected} == {'side'}
    assert [(m.kind, m.state) for m in selected] == [
        (kind, state) for state in ('rest', 'mid', 'limit') for kind in ('reference', 'candidate')]
    assert table == [[0], [0], [0]]
    assert audit['selected_view'] == 'side'


def test_invisible_defect_and_missing_or_stale_composite_audit_are_rejected(tmp_path):
    case, members, labels = _case(tmp_path, visible=False)
    with pytest.raises(ValueError, match='no qualified single-view'):
        prepare_pairs(case, members, labels)
    case, members, labels = _case(tmp_path, composite=True)
    for audits in ({}, {'hard-test': {'views': {'side': {'reference_sha': 'stale'}}}}):
        with pytest.raises(ValueError, match='hash-bound per-label'):
            prepare_pairs(case, members, labels, exposure_audits=audits)
