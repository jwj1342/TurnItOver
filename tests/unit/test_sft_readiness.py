from scripts.audit_episode_export import _action_learnability


def test_privileged_action_audit_reports_observable_conflicts():
    conflicts = [{"observable_key": "same-input", "distinct_actions": 2}]
    assert _action_learnability(conflicts, 0).startswith("unsafe:")
    assert "conflicting" in _action_learnability(conflicts, 3)


def test_privileged_action_audit_does_not_overclaim_singletons():
    assert _action_learnability([], 2).startswith("unresolved:")
    assert _action_learnability([], 0).startswith("no exact-match")
