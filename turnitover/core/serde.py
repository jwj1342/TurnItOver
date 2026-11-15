"""JSON (de)serialization for frozen dataclasses, enums, tuples and NewTypes.

Kept deliberately small: nested dataclasses, tuples/lists, dicts, enums,
Optional and Union of dataclasses (discriminated by a ``__type__`` field).
"""
from __future__ import annotations

import dataclasses
import enum
import types
import typing
from typing import Any, get_args, get_origin, get_type_hints

_TYPE_KEY = "__type__"


def to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        out[_TYPE_KEY] = type(obj).__name__
        return out
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    return obj


def from_dict(tp: Any, data: Any) -> Any:
    """Build a value of type ``tp`` from plain JSON data."""
    origin = get_origin(tp)
    if origin in (types.UnionType, typing.Union):
        return _from_union(get_args(tp), data)
    if data is None:
        if tp in (Any, type(None)):
            return None
        raise ValueError(f"Null is not allowed for {tp}")
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if _TYPE_KEY in data and data[_TYPE_KEY] != tp.__name__:
            raise ValueError(f"Unexpected type tag for {tp.__name__}: {data[_TYPE_KEY]!r}")
        hints = get_type_hints(tp)
        kwargs = {
            f.name: from_dict(hints[f.name], data[f.name])
            for f in dataclasses.fields(tp)
            if f.name in data
        }
        return tp(**kwargs)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return tp(data)
    if origin in (tuple, list):
        args = get_args(tp)
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_dict(args[0], v) for v in data)
        if origin is tuple and args:
            return tuple(from_dict(a, v) for a, v in zip(args, data, strict=True))
        elem = args[0] if args else Any
        return [from_dict(elem, v) for v in data]
    if origin is dict:
        args = get_args(tp)
        val_tp = args[1] if len(args) == 2 else Any
        return {k: from_dict(val_tp, v) for k, v in data.items()}
    return data


def _from_union(options: tuple[Any, ...], data: Any) -> Any:
    if data is None and type(None) in options:
        return None
    options = tuple(opt for opt in options if opt is not type(None))
    if len(options) == 1:
        return from_dict(options[0], data)
    if isinstance(data, dict) and _TYPE_KEY in data:
        for opt in options:
            if isinstance(opt, type) and opt.__name__ == data[_TYPE_KEY]:
                return from_dict(opt, data)
    raise ValueError(f"Expected a known {_TYPE_KEY} tag for {options}")
