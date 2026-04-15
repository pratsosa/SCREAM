import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from scream.data.transforms import get_mask_splits


# ---------------------------------------------------------------------------
# get_mask_splits
# ---------------------------------------------------------------------------

def test_masks_are_disjoint_and_cover_all():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, 5))
    train, val, test = get_mask_splits(data, train_pct=0.8)

    # No index is assigned to more than one split
    assert not np.any(train & val)
    assert not np.any(train & test)
    assert not np.any(val & test)

    # Every index is assigned to exactly one split
    assert np.all(train | val | test)


def test_split_sizes_match_train_pct():
    N = 1000
    train_pct = 0.8
    data = np.zeros((N, 3))
    train, val, test = get_mask_splits(data, train_pct=train_pct)

    expected_train = int(N * train_pct * 0.8)
    expected_val   = int(N * train_pct * 0.2)
    expected_test  = N - int(N * train_pct)

    assert train.sum() == expected_train
    assert val.sum()   == expected_val
    assert test.sum()  == expected_test


def test_reproducible_with_seed():
    data = np.zeros((100, 2))
    np.random.seed(7)
    t1, v1, _ = get_mask_splits(data, train_pct=0.8)
    np.random.seed(7)
    t2, v2, _ = get_mask_splits(data, train_pct=0.8)
    assert np.array_equal(t1, t2)
    assert np.array_equal(v1, v2)


# ---------------------------------------------------------------------------
# StandardScaler round-trip (used in datamodules)
# ---------------------------------------------------------------------------

def test_standard_scaler_roundtrip():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 9))
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    X_recovered = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X_recovered, X, atol=1e-10)
