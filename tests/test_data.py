import numpy as np
import pytest

from fact.data import class_weights, partition


def labels(n=5000, num_classes=10):
    return np.random.RandomState(0).randint(num_classes, size=n)


@pytest.mark.parametrize("alpha", [None, 0.3, 0.6])
def test_partition_is_disjoint_sized_and_deterministic(alpha):
    targets = labels()
    sizes = [700, 300, 1000, 1]
    shards = partition(targets, sizes, seed=3, dirichlet_alpha=alpha)

    assert [len(shard) for shard in shards] == sizes
    used = np.concatenate(shards)
    assert len(np.unique(used)) == len(used)
    assert used.min() >= 0 and used.max() < len(targets)
    for shard, again in zip(shards, partition(targets, sizes, seed=3, dirichlet_alpha=alpha)):
        np.testing.assert_array_equal(shard, again)


@pytest.mark.parametrize("alpha", [None, 0.3])
def test_partition_can_use_the_whole_training_set(alpha):
    targets = labels(n=2000)
    shards = partition(targets, [125] * 16, seed=0, dirichlet_alpha=alpha)
    np.testing.assert_array_equal(np.sort(np.concatenate(shards)), np.arange(2000))


def test_single_agent_gets_its_contribution():
    (shard,) = partition(labels(), [4000], seed=0)
    assert len(np.unique(shard)) == 4000


def test_dirichlet_partition_skews_labels():
    targets = labels(n=20000)

    def majority_share(alpha):
        shards = partition(targets, [1000] * 10, seed=0, dirichlet_alpha=alpha)
        return np.mean([np.bincount(targets[s], minlength=10).max() / len(s) for s in shards])

    assert majority_share(None) < 0.2
    assert majority_share(0.05) > 0.5


def test_partition_rejects_more_data_than_exists():
    with pytest.raises(ValueError, match="training set has 100"):
        partition(labels(n=100), [60, 60], seed=0)


def test_class_weights():
    np.testing.assert_allclose(class_weights(np.array([0, 0, 0, 1])), [0.25, 0.75])
