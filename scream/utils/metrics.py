def count_parameters(model) -> int:
    """Return the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())
