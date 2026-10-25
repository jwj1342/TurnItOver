"""Strict parsing for fixed-observation stopping predictions."""
from turnitover.verifier.contracts import parse_json


def parse_stopping(value):
    if isinstance(value, str):
        value = parse_json(value)
    if not isinstance(value, dict) or set(value) != {"stop_decision"}:
        raise ValueError("Return exactly one stop_decision")
    row = value["stop_decision"]
    if not isinstance(row, dict) or set(row) != {"decision", "outcome", "reason"}:
        raise ValueError("Invalid stop_decision fields")
    if not isinstance(row["reason"], str) or not row["reason"].strip():
        raise ValueError("Stopping reason must be nonempty")
    if row["decision"] == "continue":
        if row["outcome"] is not None:
            raise ValueError("Continue cannot have an outcome")
    elif row["decision"] == "stop":
        if row["outcome"] not in ("fail", "uncertain", "pass"):
            raise ValueError("Stop requires an outcome")
    else:
        raise ValueError("Unknown stopping decision")
    return row
