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
