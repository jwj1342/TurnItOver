import io
import json

import pytest

from turnitover.models.streaming import read_chat_stream


def frame(delta=None, finish=None, **kwargs):
    return {'model': 'kimi-k3', 'choices': [{'index': 0, 'delta': delta or {},
                                           'finish_reason': finish}], **kwargs}


def stream(*events, done=True):
    wire = ': heartbeat\r\n\r\n'
    for value in events:
        wire += 'data: ' + json.dumps(value) + '\r\n\r\n'
    if done:
        wire += 'data: [DONE]\r\n\r\n'
    return io.BytesIO(wire.encode())


def test_reasoning_content_and_usage_are_separated():
    usage = {'prompt_tokens': 100, 'completion_tokens': 20,
             'completion_tokens_details': {'reasoning_tokens': 15}}
    result = read_chat_stream(stream(frame({'role': 'assistant'}),
        frame({'reasoning_content': 'private thought'}), frame({'reasoning': 'more'}),
        frame({'content': '{"ok":'}), frame({'content': 'true}'}), frame(finish='stop'),
        {'model': 'kimi-k3', 'choices': [], 'usage': usage}))
    assert result['choices'][0]['message']['content'] == '{"ok":true}'
    assert result['usage'] == usage
    assert result['stream_metadata']['reasoning_characters'] == 19
    assert 'private thought' not in json.dumps(result)


@pytest.mark.parametrize('events,done', [
    ([frame({'content': 'partial'})], False),
    ([frame({'content': 'partial'}), frame(finish='stop')], False),
    ([frame({'content': 'partial'})], True),
    ([frame({'content': 'partial'}), frame(finish='length')], True),
    ([frame({'content': 'partial'}), frame(finish='tool_calls')], True),
    ([frame({'reasoning_content': 'secret'}), frame(finish='stop')], True),
    ([frame({'content': 'ok'}), frame(finish='stop'), frame({'content': 'extra'})], True),
])
def test_truncation_and_abnormal_finish_never_return_partial(events, done):
    with pytest.raises(ValueError):
        read_chat_stream(stream(*events, done=done))


@pytest.mark.parametrize('wire', [
    b'event: error\ndata: {"message":"secret-key"}\n\n',
    b'data: {"error":{"message":"secret-key"}}\n\n',
    b'data: not-json-secret-key\n\n', b'data: \xff\n\n',
])
def test_provider_errors_are_redacted(wire):
    with pytest.raises(ValueError) as error:
        read_chat_stream(io.BytesIO(wire))
    assert 'secret-key' not in str(error.value)


def test_multiline_sse_and_final_unterminated_done():
    wire = ('id: 1\nretry: 1000\nevent: message\n'
            'data: {"model":"kimi-k3",\n'
            'data: "choices":[{"delta":{"content":"好"},"finish_reason":"stop"}]}\n\n'
            ': heartbeat\n\ndata: [DONE]')
    result = read_chat_stream(io.BytesIO(wire.encode()))
    assert result['choices'][0]['message']['content'] == '好'


def test_malformed_choice_or_usage_rejected():
    for event in ({'choices': 'bad'}, {'choices': [{'index': 1}]},
                  frame({'content': ['not-text']}), frame(usage=[])):
        with pytest.raises(ValueError):
            read_chat_stream(stream(event))


def test_client_stream_integration(monkeypatch):
    from turnitover.models.client import complete
    from turnitover.models.config import model_config
    cfg = model_config('generator', {'TIO_GENERATOR_PROVIDER':'openai_compatible',
        'TIO_GENERATOR_MODEL':'kimi-k3', 'TIO_GENERATOR_STREAM':'true'})
    class Opener:
        def open(self, request, timeout):
            body = json.loads(request.data)
            assert body['stream'] is True
            assert body['stream_options']['include_usage'] is True
            return stream(frame({'reasoning_content':'hidden'}),
                frame({'content':'{"stop":true}'}), frame(finish='stop'),
                {'choices':[], 'usage':{'total_tokens':42}})
    monkeypatch.setattr('turnitover.models.client.build_opener',lambda *a:Opener())
    result = complete(cfg, 'repair')
    assert result.text == '{"stop":true}' and result.usage['total_tokens'] == 42
    assert result.finish_reason == 'stop'


@pytest.mark.parametrize('provider,value', [('openai_compatible','invalid'),('openai','true')])
def test_stream_setting_rejects_invalid_or_unsupported(provider,value):
    from turnitover.models.config import model_config
    with pytest.raises(ValueError):
        model_config('generator',{'TIO_GENERATOR_PROVIDER':provider,'TIO_GENERATOR_STREAM':value})
