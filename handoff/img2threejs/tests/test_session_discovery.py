from pathlib import Path

from src.parser import discover_sessions


def test_discover_sessions_supports_grouped_directories(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    session_a = root / "human_demo" / "session-a"
    session_b = root / "human_demo" / "nested" / "session-b"
    session_a.mkdir(parents=True)
    session_b.mkdir(parents=True)
    (session_a / "session.json").write_text('{"messages": []}', encoding="utf-8")
    (session_b / "session.json").write_text('{"messages": []}', encoding="utf-8")

    assert discover_sessions(root) == [session_a, session_b]
