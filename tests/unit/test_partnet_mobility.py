from types import SimpleNamespace

import numpy as np

from scripts.convert_partnet_mobility import _geometry_fingerprint
from scripts.download_partnet_mobility import select_files


def test_partnet_selection_is_deterministic_and_respects_size_cap():
    rows = [SimpleNamespace(rfilename="3.zip", size=10),
            SimpleNamespace(rfilename="1.zip", size=5),
            SimpleNamespace(rfilename="2.zip", size=20),
            SimpleNamespace(rfilename="README.md", size=1)]
    first = select_files(rows, count=2, seed=7, max_bytes=10)
    second = select_files(list(reversed(rows)), count=2, seed=7, max_bytes=10)
    assert [row.rfilename for row in first] == [row.rfilename for row in second]
    assert {row.rfilename for row in first} == {"1.zip", "3.zip"}


def test_geometry_fingerprint_ignores_vertex_order_and_duplicate_vertices():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    reordered = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    assert _geometry_fingerprint(vertices) == _geometry_fingerprint(reordered)
