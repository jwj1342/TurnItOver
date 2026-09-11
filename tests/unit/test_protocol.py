import base64

import numpy as np

from turnitover.render import protocol as P


def test_decode_typed_arrays():
    pos = np.arange(9, dtype="<f4")
    idx = np.array([0, 1, 2, 0, 2, 1], dtype="<u4")
    v = P.decode_positions(base64.b64encode(pos.tobytes()).decode())
    f = P.decode_indices(base64.b64encode(idx.tobytes()).decode())
    assert v.shape == (3, 3) and f.shape == (2, 3)
    assert f.tolist() == [[0, 1, 2], [0, 2, 1]]


def test_stats_keys_match_ts_protocol():
    ts = (P.__file__.rsplit("turnitover", 1)[0] + "web/src/protocol.ts")
    text = open(ts, encoding="utf-8").read()
    block = text[text.index("interface StatsPayload"):]
    block = block[: block.index("}")]
    for key in P.STATS_KEYS:
        assert key in block, f"{key} missing from web/src/protocol.ts StatsPayload"
