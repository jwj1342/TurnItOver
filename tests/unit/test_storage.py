import numpy as np

from tests.unit.test_serde import _sample
from turnitover.checkers.base import PartMesh
from turnitover.storage.reader import iter_samples, load_geometry, read_index
from turnitover.storage.writer import ShardWriter


def test_write_read_round_trip(tmp_path):
    geometry = {"rest": {"door": PartMesh(np.zeros((3, 3), np.float32), np.array([[0, 1, 2]], np.uint32))}}
    with ShardWriter(tmp_path, 2) as w:
        for i in range(3):
            s = _sample()
            s = type(s)(**{**s.__dict__, "sample_id": f"toy-{i}", "idx": i})
            w.add(s, {"obs/00.png": b"png" + bytes([i])}, geometry)
        w.write_manifest({"status": "complete"})
    assert (tmp_path / "shard-0002.tar").exists()
    rows = read_index(tmp_path / "shard-0002.index.jsonl")
    assert [r["idx"] for r in rows] == [0, 1, 2]
    assert rows[0]["defect_ids"] == ["structure.part_offset"]
    got = list(iter_samples(tmp_path / "shard-0002.tar"))
    assert [s.sample_id for s, _ in got] == ["toy-0", "toy-1", "toy-2"]
    assert got[1][1]["obs/00.png"] == b"png\x01"
    geo = load_geometry(got[0][1]["geometry.npz"])
    assert geo["rest"]["door"].faces.tolist() == [[0, 1, 2]]
