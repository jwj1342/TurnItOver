"""Offline integration tests run a REAL SDK Agent/Conversation and task tools."""
import json

import pytest

pytest.importorskip('openhands.sdk')

from litellm import ModelResponse
from openhands.sdk import LLM, Message
from openhands.sdk.llm import LLMResponse
from openhands.sdk.llm.message import MessageToolCall

from turnitover.models.client import ModelResult
from turnitover.repair.openhands_runner import ProgramTool, run_openhands_repair
from tests.unit.test_table1_protocol import observations, verdict


def mock_llm(monkeypatch, actions):
    seen = []
    def generate(self, messages, tools=None, **kwargs):
        number = len(seen)
        seen.append([message.model_dump(mode='json') for message in messages])
        assert 'PRIVATE_SENTINEL' not in json.dumps(seen[-1])
        assert {tool.name for tool in tools} == {ProgramTool.name, 'finish'}
        name, args = actions[number] if number < len(actions) else ('finish', {'message': 'done'})
        response_id = f'offline-{number}'
        self.metrics.add_token_usage(3, 4, 0, 0, 128000, response_id)
        message = Message(role='assistant', content=[], tool_calls=[MessageToolCall(
            id=f'call-{number}', name=name, arguments=json.dumps(args), origin='completion')])
        return LLMResponse(message=message, metrics=self.metrics.get_snapshot(),
                           raw_response=ModelResponse(id=response_id))
    monkeypatch.setattr(LLM, 'generate', generate)
    llm = LLM(model='openai/gpt-4o', api_key='offline-placeholder', num_retries=0,
              capability_overrides={'supports_vision': True})
    return llm, seen


def test_sdk_executes_edits_and_reverifies_current_program(tmp_path, monkeypatch, observations):
    pairs, runtime, record = observations
    sources = []
    def observe(source, folder):
        sources.append(source)
        return {'pairs': pairs, 'runtime': runtime, 'public_record': record, 'gate': {'passed': True}}
    def judge(prompt, images):
        status = 'pass' if sources[-1] == 'fixed' else 'fail'
        return ModelResult(json.dumps({'verdict': verdict(status, step=0)}), 'frozen',
                           {'total_tokens': 5}, 'stop', 1)
    llm, seen = mock_llm(monkeypatch, [
        (ProgramTool.name, {'operation': 'read'}),
        (ProgramTool.name, {'operation': 'submit', 'edits': [{'old': 'broken', 'new': 'fixed'}]})])
    result = run_openhands_repair('broken', None, observe,
        lambda source: {'passed': source == 'fixed', 'private': 'PRIVATE_SENTINEL'},
        tmp_path/'run', condition='evidence', max_rounds=2, max_agent_steps=5,
        judge_model_call=judge, llm=llm)
    assert sources == ['broken', 'fixed']
    assert result['termination'] == 'judge_pass'
    assert result['harness'] == 'openhands-sdk'
    assert result['generator_calls'] == len(seen) == 2
    assert result['completed_rounds'] == result['first_success_round'] == 1
    assert result['final_passed'] is True
    assert (tmp_path/'run/final.ts').read_text() == 'fixed'
    assert any((tmp_path/'run/conversation').rglob('*.json'))
    assert result['tokens_to_first_success'] == 44
    assert result['total_tokens'] == 44


def test_compatible_generator_capabilities_are_explicit_and_frozen():
    from turnitover.models.config import model_config
    from turnitover.repair.openhands_runner import generator_llm
    config = model_config('generator', {
        'TIO_GENERATOR_PROVIDER': 'openai_compatible',
        'TIO_GENERATOR_MODEL': 'kimi-k3',
        'TIO_GENERATOR_BASE_URL': 'https://api.moonshot.cn/v1',
        'TIO_GENERATOR_SUPPORTS_VISION': 'true'})
    llm = generator_llm(config)
    assert llm.model == 'openai/kimi-k3'
    assert llm.base_url == config.base_url
    assert llm.vision_is_active()
    assert config.public()['supports_vision'] is True
    with pytest.raises(ValueError, match='SUPPORTS_VISION'):
        model_config('generator', {'TIO_GENERATOR_SUPPORTS_VISION': 'automatic'})


def test_sdk_none_finish_consumes_unchanged_round_until_fixed_budget(tmp_path, monkeypatch):
    llm, seen = mock_llm(monkeypatch, [])
    result = run_openhands_repair('source', None,
        lambda *args: {'pairs': [], 'runtime': [], 'public_record': None},
        lambda source: {'passed': None, 'private': 'PRIVATE_SENTINEL'}, tmp_path/'none',
        condition='none', max_rounds=3, max_agent_steps=5, llm=llm)
    assert result['completed_rounds'] == len(seen) == 3
    assert result['termination'] == 'round_limit'
    assert result['final_passed'] is None


def test_sdk_private_audit_exception_does_not_become_tool_feedback(tmp_path, monkeypatch):
    llm, seen = mock_llm(monkeypatch, [
        (ProgramTool.name, {'operation': 'submit', 'edits': [{'old': 'source', 'new': 'edited'}]})])
    def audit(source):
        if source == 'edited':
            raise ValueError('PRIVATE_SENTINEL')
        return {'passed': False}
    with pytest.raises(ValueError, match='PRIVATE_SENTINEL'):
        run_openhands_repair('source', None, lambda *a: {'gate': {'passed': True}},
            audit, tmp_path/'failure', condition='none', max_rounds=2, max_agent_steps=5, llm=llm)
    assert len(seen) == 1
    result = json.loads((tmp_path/'failure/result.json').read_text())
    assert result['status'] == 'incomplete'
    events = ''.join(path.read_text(encoding='utf-8') for path in (tmp_path/'failure/conversation').rglob('*.json'))
    assert 'PRIVATE_SENTINEL' not in events


def test_sdk_global_step_cap_survives_none_resumes(tmp_path, monkeypatch):
    llm, seen = mock_llm(monkeypatch, [])
    result = run_openhands_repair('source', None, lambda *a: {}, lambda source: {'passed': False},
        tmp_path/'limited', condition='none', max_rounds=3, max_agent_steps=2, llm=llm)
    assert len(seen) == 2
    assert result['termination'] == 'agent_step_limit'
    assert result['completed_rounds'] == 2


def test_batch_preserves_fresh_runtime_through_real_sdk(tmp_path, monkeypatch, observations):
    from scripts.run_rp29_repair_batch import run_case
    from turnitover.oracle.supervision_validation import public_payload
    import turnitover.repair.openhands_runner as runner
    pairs, _, record = observations
    source = tmp_path/'initial.ts'
    source.write_text('broken', encoding='utf-8')
    case = {'id': 'case', 'initial_program': source, 'reference_image': pairs[0].reference,
            'public_context': public_payload(record)['context']}
    llm, seen = mock_llm(monkeypatch, [(ProgramTool.name, {'operation': 'submit',
        'edits': [{'old': 'broken', 'new': 'fixed'}]})])
    monkeypatch.setattr(runner, 'generator_llm', lambda config: llm)
    current = []
    def replay_factory(*args):
        def replay(source):
            current.append(source)
            return {'pairs': [{'state': pair.state, 'reference': str(pair.reference),
                              'candidate': str(pair.candidate)} for pair in pairs],
                    'runtime': [{'state': 'mid', 'kind': 'public_stats',
                                 'value': {'current_source': source}}], 'gate': {'passed': True},
                    'private_evidence': {'candidate': {'path': 'PRIVATE_SENTINEL',
                                                       'geometry': 'PRIVATE_SENTINEL'}}}
        return replay
    def judge(prompt, images):
        payload = json.loads(prompt.rsplit('\n', 1)[-1])
        runtime = [row['payload'] for row in payload['history'] if row['action']['type'] == 'query_runtime']
        if payload['history'][0]['state'] == 'mid':
            assert runtime[0]['value']['current_source'] == current[-1]
        return ModelResult(json.dumps({'verdict': verdict('pass' if current[-1] == 'fixed' else 'fail', step=0)}),
                           'frozen', {'total_tokens': 5}, 'stop', 1)
    result = run_case(tmp_path/'manifest.json', case, tmp_path, tmp_path/'batch', None,
        'evidence', 1, None, audit_factory=lambda *args: lambda source: {'passed': source == 'fixed'},
        replay_factory=replay_factory, max_agent_steps=3, judge_model_call=judge)
    assert result['status'] == 'complete'
    assert result['final_passed'] is True
    assert current == ['broken', 'fixed']
    assert len(seen) == 1


def test_sdk_compile_failure_feedback_allows_a_later_repair(tmp_path, monkeypatch):
    from turnitover.repair.table1_scoring import make_repair_audit
    import hashlib
    def replay(source):
        return {'status': 'complete', 'program_sha256': {
            'candidate': hashlib.sha256(source.encode()).hexdigest()},
            'gate': {'passed': source != 'broken', 'stage': 'compile',
                     'diagnostic': 'Expected semicolon at line 1', 'private': 'PRIVATE_SENTINEL'}}
    audit = make_repair_audit(replay, [{'defect_id': 'structure.part_offset'}],
        {'structure.part_offset': lambda label, ev: {'passed': True, 'checker_id': 'test'}},
        {'compile_render': lambda _: True}, ['compile_render'])
    llm, seen = mock_llm(monkeypatch, [
        (ProgramTool.name, {'operation': 'submit', 'edits': [{'old': 'source', 'new': 'broken'}]}),
        (ProgramTool.name, {'operation': 'submit', 'edits': [{'old': 'broken', 'new': 'fixed'}]})])
    result = run_openhands_repair('source', None, lambda source, _: replay(source), audit,
        tmp_path/'compile', condition='none', max_rounds=2, max_agent_steps=4, llm=llm)
    assert len(seen) == 2
    assert 'Expected semicolon' in json.dumps(seen[1])
    assert result['final_passed'] is True
    assert result['completed_rounds'] == 2
