"""Exact transactional edits against the entire current TypeScript program, with geometry arrays elided."""
from __future__ import annotations

import re

GEOMETRY = re.compile(r'\[[-0-9.eE,\s]{200,}\]')
PLACEHOLDER = re.compile(r'__GEOM_\d+__')


def elide(source: str) -> tuple[str, list[str]]:
    """Replace each numeric array literal of 200+ characters with __GEOM_<k>__, k in order of appearance."""
    arrays = []
    def placeholder(match: re.Match) -> str:
        arrays.append(match.group())
        return f'__GEOM_{len(arrays)-1}__'
    return GEOMETRY.sub(placeholder, source), arrays


def expand(text: str, arrays: list[str]) -> str:
    return PLACEHOLDER.sub(lambda match: arrays[int(match.group()[7:-2])], text)


def apply_response(source: str, response: dict) -> str | None:
    """Return None for a stop decision; never partially apply an ambiguous patch.
    Placeholders in source are fixed: no edit may overlap one or write placeholder text."""
    if not isinstance(response, dict):
        raise ValueError('Repair response must be an object')
    if set(response) == {'stop'} and response['stop'] is True:
        return None
    if set(response) != {'edits'} or not isinstance(response['edits'], list) or not 1 <= len(response['edits']) <= 32:
        raise ValueError('Expected stop=true or 1..32 exact edits')
    placeholders = [match.span() for match in PLACEHOLDER.finditer(source)]
    spans = []
    for edit in response['edits']:
        if not isinstance(edit, dict) or set(edit) != {'old', 'new'}:
            raise ValueError('Each edit requires exactly old/new fields')
        old, new = edit['old'], edit['new']
        if not isinstance(old, str) or not old or not isinstance(new, str) or old == new:
            raise ValueError('Edit must replace nonempty text with different text')
        start = source.find(old)
        if start < 0 or source.find(old, start+1) >= 0:
            raise ValueError('Each old string must occur exactly once in the current program')
        if '__GEOM_' in new or any(start < right and left < start+len(old) for left, right in placeholders):
            raise ValueError('Edits must not touch, remove, duplicate or move __GEOM_<k>__ placeholders')
        spans.append((start, start+len(old), new))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError('Edits overlap in the original source')
    result = source
    for start, end, new in reversed(spans):
        result = result[:start]+new+result[end:]
    return result
