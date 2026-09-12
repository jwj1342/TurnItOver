"""Exact transactional edits against the entire current TypeScript program."""
from __future__ import annotations


def apply_response(source, response):
    """Return None for a stop decision; never partially apply an ambiguous patch."""
    if not isinstance(response, dict):
        raise ValueError('Repair response must be an object')
    if set(response) == {'stop'} and response['stop'] is True:
        return None
    if set(response) != {'edits'} or not isinstance(response['edits'], list) or not 1 <= len(response['edits']) <= 32:
        raise ValueError('Expected stop=true or 1..32 exact edits')
    spans=[]
    for edit in response['edits']:
        if not isinstance(edit, dict) or set(edit) != {'old','new'}:
            raise ValueError('Each edit requires exactly old/new fields')
        old,new=edit['old'],edit['new']
        if not isinstance(old,str) or not old or not isinstance(new,str) or old==new:
            raise ValueError('Edit must replace nonempty text with different text')
        start=source.find(old)
        if start<0 or source.find(old,start+1)>=0:
            raise ValueError('Each old string must occur exactly once in the current program')
        spans.append((start,start+len(old),new))
    spans.sort()
    if any(left[1]>right[0] for left,right in zip(spans,spans[1:])):
        raise ValueError('Edits overlap in the original source')
    result=source
    for start,end,new in reversed(spans):
        result=result[:start]+new+result[end:]
    return result
