"""Behavior checks for budget, feedback isolation and the portable release boundary."""
from pathlib import Path
import json
import sys
import subprocess
from types import SimpleNamespace
import pytest

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
sys.path[:0] = [str(PKG), str(PKG / 'frozen'), str(REPO)]
import repair
from turnitover.repair import __path__ as repair_path
repair_path.insert(0, str(PKG / 'frozen/turnitover/repair'))


@pytest.fixture
def runner():
    pytest.importorskip('openhands.sdk')
    from turnitover.repair import openhands_runner
    return openhands_runner


def task(runner, tmp_path, condition='none'):
    pairs = [SimpleNamespace(state=f'state-{i}', reference=Path(f'reference-{i}.png'), candidate=Path(f'candidate-{i}.png')) for i in range(3)]
    observe = lambda source, folder: {'pairs': pairs, 'public_record': {'public': True}}
    audit = lambda source: {'usable': True, 'passed': False, 'private_parameter_updates': {'secret_reference': 123}}
    return runner.RepairTask('synthetic program', observe, audit, tmp_path, condition, 3, None, None, None, 'Repair observed differences.', None)


def test_frozen_sources_and_runtime_match():
    repair.check_sources(REPO)


def test_input_path_escape_is_rejected(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    (tmp_path / 'outside.txt').write_text('not a dataset input')
    with pytest.raises(ValueError, match='out-of-root'):
        repair.inside(root, '../outside.txt')


def test_program_view_respects_serialized_budget_and_query(runner):
    source = ('\\"\n' * 20000) + 'TARGET' + ('x' * 50000)
    view = runner.program_view(source, query='TARGET')
    assert len(json.dumps(view, ensure_ascii=False)) <= 45000
    assert 'TARGET' in view['current_program']
    assert view['source_incomplete'] is True
    assert view['current_program'] == source[view['offset']:view['end_offset']]


def test_none_feedback_has_all_images_and_no_private_audit(runner, tmp_path):
    t = task(runner, tmp_path)
    t.check(tmp_path, 0)
    public, images = t.feedback(tmp_path)
    assert len(images) == len(public['image_order']) == 6
    assert not {'status', 'findings', 'passed', 'private_parameter_updates'} & public.keys()
    assert 'secret_reference' not in json.dumps(public)
    assert t.termination is None


def test_three_empty_submissions_consume_budget(runner, tmp_path):
    t = task(runner, tmp_path)
    for _ in range(3):
        t.submit([])
    assert t.report['completed_rounds'] == 3
    assert t.termination == 'round_limit'
    assert t.submit([]) == ({'status': 'budget_exhausted'}, ())
    assert len(list(tmp_path.glob('round-*'))) == 3


def test_public_pass_stops_but_does_not_make_private_success(runner, tmp_path, monkeypatch):
    t = task(runner, tmp_path, condition='pixel')
    monkeypatch.setattr(runner, 'evaluate_judge', lambda *a, **k: {'status': 'pass', 'feedback': {'summary': 'No observed difference'}, 'observations': []})
    t.check(tmp_path, 0)
    t.feedback(tmp_path)
    t.record_success(0)
    assert t.termination == 'judge_pass'
    assert t.current_audit['passed'] is False
    assert t.report['first_success_round'] is None


@pytest.mark.parametrize('audit', [{'usable': False, 'passed': True}, {'usable': True}, {'usable': True, 'passed': 'true'}])
def test_invalid_private_audit_cannot_pass(runner, tmp_path, audit):
    t = task(runner, tmp_path)
    t.audit = lambda source: audit
    with pytest.raises(ValueError):
        t.check(tmp_path, 0)


def test_unusable_physical_program_retains_unknown_findings():
    from physical_scoring_v4 import score_original
    labels = [{'defect_id': 'kinematics.joint_axis', 'parts': ['synthetic-part']}]
    rows = score_original('unsupported', 'reference', labels, usable=False)
    assert len(rows) == 1 and rows[0]['resolved'] is None
    assert rows[0]['defect_id'] == labels[0]['defect_id']


def test_prepare_never_overwrites_work(tmp_path):
    args = SimpleNamespace(runtime=REPO, out=tmp_path)
    with pytest.raises(ValueError, match='new directory'):
        repair.prepare(args)


@pytest.mark.parametrize('payload,expected', [({'observations': []}, 0), ({'annotation': {'synthetic': True}}, 1)])
def test_input_audit_rejects_private_feedback(tmp_path, payload, expected):
    folder = tmp_path / 'work/runs-v2/none/rep-1/synthetic-case'
    (folder / 'conversation').mkdir(parents=True)
    (folder / 'result.json').write_text(json.dumps({'status': 'complete', 'completed_rounds': 1, 'generator_calls': 1}))
    (folder / 'observation-manifest.json').write_text(json.dumps({'generator_images': [{}, {}]}))
    event = {'content': [{'text': json.dumps(payload)}, {'image_urls': ['synthetic-reference', 'synthetic-candidate']}], 'reasoning_tokens': 0}
    (folder / 'conversation/event.json').write_text(json.dumps(event))
    result = subprocess.run([sys.executable, str(PKG / 'audit_inputs.py'), '--work', str(tmp_path / 'work'), '--out', str(tmp_path / 'audit')], capture_output=True, text=True)
    assert result.returncode == expected, result.stderr
    report = json.loads((tmp_path / 'audit/summary.json').read_text())
    assert len(report['anomalies']) == expected


@pytest.mark.parametrize('status', [None, 'execution_error', 'complete'])
def test_incomplete_trial_is_preserved_before_any_api_call(tmp_path, status):
    trial = {'case_id': 'synthetic-case', 'repetition': 1}
    trial_path = tmp_path / 'trials.private.json'
    trial_path.write_text(json.dumps([trial]))
    folder = tmp_path / 'runs-v2/none/rep-1/synthetic-case'
    folder.mkdir(parents=True)
    if status:
        (folder / 'result.json').write_text(json.dumps({'status': status}))
    if status == 'complete':
        assert repair.checked_trials(tmp_path, 'none', trial_path) == [('synthetic-case', 1)]
    else:
        with pytest.raises(ValueError, match='Incomplete attempt preserved'):
            repair.checked_trials(tmp_path, 'none', trial_path)
    assert folder.exists()
