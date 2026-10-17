from pathlib import Path

import pytest

from turnitover import cli


def test_verify_and_iterate_share_verification_defaults(monkeypatch, tmp_path):
    seen = {}

    def verify(args):
        seen["verify"] = args
        return 0

    def iterate(args):
        seen["iterate"] = args
        return 0

    monkeypatch.setattr(cli, "_verify", verify)
    monkeypatch.setattr(cli, "_iterate", iterate)

    assert cli.main(["verify", "--program", "candidate.ts", "--out", "verify-out"]) == 0
    assert cli.main(["iterate", "--reference-image", "reference.png", "--out", "iterate-out"]) == 0

    verify_args, iterate_args = seen["verify"], seen["iterate"]
    assert verify_args.reference_image == []
    assert iterate_args.reference_image == [Path("reference.png")]
    for name in ("task", "policy", "budget", "seed", "size", "views", "actions", "max_triangles",
                 "max_draw_calls", "env_file"):
        assert getattr(verify_args, name) == getattr(iterate_args, name)

    with pytest.raises(SystemExit):
        cli.main(["iterate", "--out", str(tmp_path / "missing-reference")])
