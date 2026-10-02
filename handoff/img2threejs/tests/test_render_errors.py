from src.runtime.errors import classify_render_failure


def test_classifies_missing_canvas() -> None:
    failure = classify_render_failure(
        page_loaded=True,
        canvas_found=False,
        canvas_nonempty=False,
        load_errors=[],
        page_errors=[],
        console_errors=[],
    )
    assert failure is not None
    assert failure.kind == "missing_canvas"


def test_classifies_clean_render_as_none() -> None:
    failure = classify_render_failure(
        page_loaded=True,
        canvas_found=True,
        canvas_nonempty=True,
        load_errors=[],
        page_errors=[],
        console_errors=[],
    )
    assert failure is None
