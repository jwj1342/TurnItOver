import importlib.util
import json
from pathlib import Path

import pytest

from turnitover.models.client import ModelResult
from turnitover.repair.table1_runner import run_table1_repair


def module():
    spec = importlib.util.spec_from_file_location('batch', Path(__file__).parents[2] / 'scripts/run_rp29_repair_batch.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_unknown_audit_never_becomes_success(tmp_path):
    result = run_table1_repair('source', lambda *a: ModelResult('{"stop":true}', 'model',
        {'total_tokens': 3}, 'stop', 1), None, lambda source: {'passed': None},
        tmp_path / 'run', max_rounds=2, allow_unscored=True)
    assert result['status'] == 'complete'
    assert result['final_passed'] is None
    assert result['first_success_round'] is None
    assert result['tokens_to_first_success'] is None
    assert result['total_tokens'] == 6
    with pytest.raises(ValueError, match='boolean'):
        run_table1_repair('source', None, None, lambda source: {'passed': None}, tmp_path / 'strict')


def test_batch_keeps_failed_denominator_and_resume_never_repeats_calls(tmp_path, monkeypatch):
    batch = module()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    cases = [(tmp_path / f'{name}.json', {'id': name,
              'private_labels': [{'defect_id': 'structure.part_offset'}]}) for name in ('a', 'b')]
    monkeypatch.setattr(batch, 'load_cases', lambda p: ({'benchmark': 'RP29 v2.1-pre.1'}, cases))
    calls = []
    def runner(path, case, native, folder, *args):
        calls.append(case['id'])
        if case['id'] == 'a':
            raise RuntimeError('API failed')
        folder.mkdir(parents=True)
        result = {'status': 'complete', 'total_tokens': 7, 'final_passed': None}
        batch.write(folder / 'summary.json', result)
        return result
    env = {'TIO_GENERATOR_PROVIDER': 'openai_compatible', 'TIO_GENERATOR_MODEL': 'test'}
    output = tmp_path / 'output'
    result = batch.run(manifest, tmp_path, output, env, conditions=('none',), case_runner=runner, collect_unscored=True)
    assert calls == ['a', 'b']
    assert result['status'] == 'incomplete'
    assert result['processed'] == 2
    assert len(result['results']) == 2
    assert result['success_rate'] is None
    assert result['total_tokens'] == 7
    assert result['conditions']['none']['resolved_by_type']['structure.part_offset'] == {
        'total': 2, 'known': 0, 'resolved': 0, 'rate': None}
    before = (output / 'none/b/summary.json').read_bytes()
    batch.run(manifest, tmp_path, output, env, conditions=('none',), case_runner=runner, collect_unscored=True)
    assert calls == ['a', 'b']
    assert (output / 'none/b/summary.json').read_bytes() == before
    with pytest.raises(ValueError, match='changed'):
        batch.run(manifest, tmp_path, output, env, rounds=4, conditions=('none',), case_runner=runner, collect_unscored=True)


def test_none_public_task_and_unscored_audit_do_not_leak_runtime(tmp_path):
    batch = module()
    source = tmp_path / 'initial.ts'; source.write_text('initial')
    reference = tmp_path / 'ref.png'; reference.write_bytes(b'reference')
    case = {'id': 'case', 'initial_program': source, 'reference_image': reference,
            'public_task': 'Match reference', 'task_runtime_requirements': [{'budget': 6}]}
    seen = []
    def replay_factory(*args):
        return lambda source: {'program_sha256': {'candidate': 'private-sha'},
                                'runtime': 'PRIVATE_RUNTIME', 'pairs': []}
    def generator(prompt, images):
        assert 'PRIVATE_RUNTIME' not in prompt
        assert 'private-sha' not in prompt
        assert 'Match reference' in prompt
        assert 'budget' in prompt
        seen.append(prompt)
        return ModelResult('{"stop":true}', 'model', {'total_tokens': 4}, 'stop', 1)
    result = batch.run_case(tmp_path/'manifest.json', case, tmp_path, tmp_path/'out', generator,
                            'none', 2, None, replay_factory=replay_factory, harness='legacy')
    assert len(seen) == 2
    assert result['final_passed'] is None
    assert result['total_tokens'] == 8


def test_scoring_gate_blocks_before_any_model_setup(tmp_path, monkeypatch):
    batch = module()
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    monkeypatch.setattr(batch, 'load_cases', lambda p: ({}, [(manifest, {'id': 'case'})]))
    monkeypatch.setattr(batch, 'model_config', lambda *a: pytest.fail('Model setup before scoring gate'))
    result = batch.run(manifest, tmp_path, tmp_path/'out', {}, conditions=('none',), preflight_only=True)
    assert result['status'] == 'inputs_validated' and result['api_calls'] == 0
    assert any('--rounds' in blocker for blocker in result['blockers'])
    assert any('exposure' in blocker for blocker in result['blockers'])
    assert not (tmp_path/'out').exists()
    with pytest.raises(ValueError, match='Full independent scoring unavailable'):
        batch.run(manifest, tmp_path, tmp_path/'out', {}, conditions=('none',))
    assert not (tmp_path/'out').exists()


def test_reject_underfilled_manifest(tmp_path):
    batch = module()
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps({'status':'frozen', 'split':'validation',
        'benchmark':'RP29 v2.1-pre.1',
        'counts':{'validation_total':120,'defective':109,'excluded_clean':11}, 'cases':[]}))
    with pytest.raises(ValueError, match='109-defect'):
        batch.load_cases(path)


def test_builtin_native_evaluator_preflight_reports_actual_partial_coverage(tmp_path, monkeypatch):
    batch = module()
    manifest = tmp_path/'manifest.json'; manifest.write_text('{}')
    monkeypatch.setattr(batch, 'load_cases', lambda p: ({}, [(manifest,
        {'id': 'case', 'private_labels': [{'defect_id': 'structure.part_offset'}]})]))
    monkeypatch.setattr(batch, 'model_config', lambda *a: pytest.fail('Preflight made model setup'))
    result = batch.run(manifest, tmp_path, tmp_path/'out', {}, conditions=('none',),
        preflight_only=True, evaluator_module='turnitover.repair.native_evaluator')
    assert len(result['checker_coverage']['supported']) == 29
    assert len(result['checker_coverage']['missing']) == 0
    assert result['scoring_contract']['status'] == 'development'
    assert result['experiment_ready'] is False and result['api_calls'] == 0
    assert not (tmp_path/'out').exists()


def test_shared_output_lock_prevents_duplicate_queue(tmp_path, monkeypatch):
    batch = module()
    import hashlib, tempfile
    output = tmp_path/'out'
    key = hashlib.sha256(str(output.resolve()).encode()).hexdigest()
    path = Path(tempfile.gettempdir())/('rp29-batch-'+key+'.lock')
    monkeypatch.setattr(batch, '_run', lambda *a, **k: pytest.fail('Duplicate queue started'))
    with batch._output_lock(path):
        with pytest.raises(RuntimeError, match='Another process'):
            batch.run(tmp_path/'manifest.json',tmp_path,output,{})


def test_missing_lazy_scoring_dependency_blocks_before_model_setup(tmp_path, monkeypatch):
    import turnitover.repair.native_evaluator as evaluator
    batch = module()
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    monkeypatch.setattr(batch, 'load_cases', lambda p: ({}, [(manifest, {'id': 'case'})]))
    monkeypatch.setattr(evaluator, 'check_environment', lambda: {'ready': False, 'missing': ['networkx']})
    monkeypatch.setattr(batch, 'model_config', lambda *a: pytest.fail('Model setup before environment check'))
    report = batch.run(manifest, tmp_path, tmp_path/'out', {}, conditions=('none',), preflight_only=True,
        evaluator_module='turnitover.repair.native_evaluator')
    assert report['api_calls'] == 0 and report['scoring_environment']['ready'] is False
    assert any('Independent scoring environment not ready' in item for item in report['blockers'])
    with pytest.raises(ValueError, match='Independent scoring environment not ready'):
        batch.run(manifest, tmp_path, tmp_path/'out', {}, conditions=('none',), rounds=3, max_agent_steps=10,
            repetition='r1', collect_unscored=True, evaluator_module='turnitover.repair.native_evaluator')
    assert not (tmp_path/'out').exists()
