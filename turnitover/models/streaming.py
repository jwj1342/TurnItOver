"""Strict Chat Completions SSE assembly, retaining the actual received wire."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import BinaryIO


class StreamingError(RuntimeError):
    """Incomplete or incompatible stream; messages never include provider payloads."""


def read_chat_stream(response: BinaryIO, wire_path: Path | None = None) -> dict:
    content, reasoning = [], []
    model, finish, usage = None, None, None
    done = False
    event = []

    def consume(lines: list[str]) -> None:
        nonlocal model, finish, usage, done
        if not lines:
            return
        payload = "\n".join(lines)
        if done:
            raise StreamingError("Provider sent data after the SSE DONE marker")
        if payload == "[DONE]":
            done = True
            return
        try:
            chunk = json.loads(payload)
        except ValueError:
            raise StreamingError("Provider returned invalid SSE JSON") from None
        if not isinstance(chunk, dict) or chunk.get("error") is not None:
            raise StreamingError("Provider returned an SSE error chunk")
        returned_model = chunk.get("model")
        if returned_model:
            if not isinstance(returned_model, str) or (model is not None and model != returned_model):
                raise StreamingError("Provider changed model identity within the SSE stream")
            model = returned_model
        if chunk.get("usage") is not None:
            if not isinstance(chunk["usage"], dict):
                raise StreamingError("Provider returned invalid SSE usage")
            usage = chunk["usage"]
        choices = chunk.get("choices", [])
        if not isinstance(choices, list):
            raise StreamingError("Provider returned invalid SSE choices")
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                raise StreamingError("Provider returned unsupported SSE choice index")
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise StreamingError("Provider returned invalid SSE delta")
            for key, pieces in (("content", content), ("reasoning_content", reasoning)):
                value = delta.get(key)
                if value is not None:
                    if not isinstance(value, str) or finish is not None:
                        raise StreamingError("Provider returned invalid or late SSE content")
                    pieces.append(value)
            reason = choice.get("finish_reason")
            if reason is not None:
                if not isinstance(reason, str) or not reason or finish is not None:
                    raise StreamingError("Provider returned invalid SSE finish reason")
                finish = reason

    with wire_path.open("wb") if wire_path is not None else nullcontext() as wire:
        for raw in response:
            if wire is not None:
                wire.write(raw)
                wire.flush()  # Preserve received evidence even when transport later fails.
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeError:
                raise StreamingError("Provider returned invalid SSE UTF-8") from None
            if not line:
                consume(event)
                event = []
            elif line.startswith("data:"):
                event.append(line[5:].removeprefix(" "))
            # Comments, event IDs and keepalive fields carry no model content.
        consume(event)
    if not done or finish is None:
        raise StreamingError("Incomplete SSE response: missing DONE marker or finish reason")
    message = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    result = {"model": model or "", "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
    if usage is not None:
        result["usage"] = usage
    return result
