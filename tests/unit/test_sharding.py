from turnitover.engine.sharding import parse_shard, sample_seed, shard_indices


def test_shards_partition_index_space():
    n, k = 100, 7
    seen = sorted(i for s in range(k) for i in shard_indices(n, s, k))
    assert seen == list(range(n))


def test_seeds_stable_and_distinct():
    seeds = [sample_seed(1234, i) for i in range(50)]
    assert len(set(seeds)) == 50
    # Pin the seed mapping so existing shards remain reproducible after refactoring.
    assert seeds[:4] == [6882349382922872486, 5790931612419668221, 3844723945885009171, 197244009799789105]
    assert sample_seed(1234, 3) != sample_seed(1235, 3)


def test_parse_shard():
    assert parse_shard("3/8") == (3, 8)
    assert parse_shard("0") == (0, 1)
