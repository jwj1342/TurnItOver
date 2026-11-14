"""Read OpenAI-compatible SSE without retaining private reasoning text."""
from __future__ import annotations

import json

_NORMAL_FINISH = frozenset({'stop', 'completed', 'end_turn', 'STOP'})


def _events(response):
    data, event = [], ''
    for raw in response:
        try:
            line = raw.decode('utf-8') if isinstance(raw, bytes) else raw
        except UnicodeDecodeError:
            raise ValueError('Invalid UTF-8 in model stream') from None
        line = line.rstrip('\r\n')
        if not line:
            if data or event:
                yield event, '\n'.join(data)
            data, event = [], ''
        elif not line.startswith(':'):
            field, _, value = line.partition(':')
            if value.startswith(' '):
                value = value[1:]
            if field == 'data':
                data.append(value)
            elif field == 'event':
                event = value
    # An unterminated final event is still parsed, but success needs [DONE].
    if data or event:
        yield event, '\n'.join(data)


def read_chat_stream(response) -> dict:
    """Return a canonical one-choice completion only after stop and [DONE].

    Reasoning deltas are consumed but only character counts are returned.
    Usage is preserved as reported by the final nonempty usage event. Error
    messages deliberately exclude provider bodies and partial model text.
    """
    content, model, finish, usage = [], '', None, {}
    chunks, reasoning_chars, done = 0, 0, False
    for event, data in _events(response):
        if event == 'error':
            raise ValueError('Provider reported a streaming error')
        if not data:
            continue
        if data.strip() == '[DONE]':
            done = True
            break
        try:
            chunk = json.loads(data)
        except (ValueError, TypeError):
            raise ValueError('Invalid JSON in model stream') from None
        if not isinstance(chunk, dict) or 'error' in chunk:
            raise ValueError('Invalid or failed model stream event')
        chunks += 1
        if chunk.get('model'):
            if not isinstance(chunk['model'], str):
                raise ValueError('Invalid model identity in stream')
            if model and model != chunk['model']:
                raise ValueError('Model identity changed during stream')
            model = chunk['model']
        if chunk.get('usage') is not None:
            if not isinstance(chunk['usage'], dict):
                raise ValueError('Invalid usage in model stream')
            if chunk['usage']:
                usage = chunk['usage']
        choices = chunk.get('choices', [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError('Expected one model stream choice')
        for choice in choices:
            if not isinstance(choice, dict) or choice.get('index', 0) != 0:
                raise ValueError('Invalid model stream choice')
            delta = choice.get('delta', {})
            if not isinstance(delta, dict):
                raise ValueError('Invalid model stream delta')
            for name in ('content', 'reasoning_content', 'reasoning'):
                text = delta.get(name)
                if text is not None and not isinstance(text, str):
                    raise ValueError('Invalid text delta in model stream')
                if text:
                    if finish is not None:
                        raise ValueError('Text received after stream finish')
                    if name == 'content':
                        content.append(text)
                    else:
                        reasoning_chars += len(text)
            reason = choice.get('finish_reason')
            if reason is not None:
                if reason not in _NORMAL_FINISH or finish is not None:
                    raise ValueError('Model stream did not finish normally')
                finish = reason
    if not done or finish is None:
        raise ValueError('Model stream truncated before normal finish and [DONE]')
    text = ''.join(content)
    if not text.strip():
        raise ValueError('Model stream returned no answer text')
    return {'model': model, 'choices': [{'index': 0,
            'message': {'role': 'assistant', 'content': text}, 'finish_reason': finish}],
            'usage': usage, 'stream_metadata': {'chunks': chunks,
                                               'reasoning_characters': reasoning_chars}}
