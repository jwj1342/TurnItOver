from scripts.tier_pilot import needs_extra_view


def _views(**fractions):
    return {v: {"core_changed_fraction": f, "passes_gate": f >= 0.025, "in_frame": True} for v, f in fractions.items()}


def test_hard_needs_unclear_default_and_clearly_better_view():
    assert needs_extra_view(False, "back", _views(back=0.031, right=0.072))
    assert not needs_extra_view(True, "back", _views(back=0.031, right=0.072))      # seen at a glance
    assert not needs_extra_view(False, "back", _views(back=0.031, right=0.050))     # no clearly better view
    assert not needs_extra_view(False, "a", _views(a=0.025, b=0.010))                # other view fails the gate
    assert not needs_extra_view(None, "back", _views(back=0.031, right=0.072))      # unreviewed is not False


def test_out_of_frame_view_does_not_count():
    views = _views(back=0.031, right=0.080)
    views["right"]["in_frame"] = False
    assert not needs_extra_view(False, "back", views)
