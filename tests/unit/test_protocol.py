import base64

import numpy as np

from turnitover.render import protocol as P


def test_decode_typed_arrays():
    pos = np.arange(9, dtype="<f4")
    idx = np.array([0, 1, 2, 0, 2, 1], dtype="<u4")
    v = P.decode_positions(base64.b64encode(pos.tobytes()).decode())
    f = P.decode_indices(base64.b64encode(idx.tobytes()).decode())
    assert v.shape == (3, 3) and f.shape == (2, 3)
    assert np.array_equal(v, pos.reshape(3, 3))
    assert f.tolist() == [[0, 1, 2], [0, 2, 1]]
