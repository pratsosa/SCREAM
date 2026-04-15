import numpy as np


def get_mask_splits(embeddings, train_pct):
    """
    Function which splits given data into Train, Validation, and Test sets.
    embeddings: Data to be split (only used in order to get the shape)
    train_pct: Percentage of the data to use for training (20% of training data will be allocated to validation)
    """
    N = embeddings.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    split_idx1, split_idx2 = int(N * train_pct * .2), int(N * train_pct)
    val_indices = indices[:split_idx1]
    train_indices = indices[split_idx1:split_idx2]
    test_indices = indices[split_idx2:]
    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)
    val_mask[val_indices] = True
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask


def get_kfold_masks(N, n_folds, fold_idx, seed):
    """
    Splits N indices into K folds and returns boolean masks for fold `fold_idx`.

    fold_idx (1/K of data) → test mask
    Of the remaining (K-1)/K, the first 20% → val mask, rest → train mask

    Returns (train_mask, val_mask, test_mask)
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)

    fold_boundaries = np.array_split(indices, n_folds)
    test_indices = fold_boundaries[fold_idx]

    remaining = np.concatenate(
        [fold_boundaries[i] for i in range(n_folds) if i != fold_idx]
    )
    val_split = int(len(remaining) * 0.2)
    val_indices = remaining[:val_split]
    train_indices = remaining[val_split:]

    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask
