from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class StreamConfig:
    name: str
    raw_data_path: str
    generated_data_path: str
    features: List[str]
    error_features: List[str]
    flow_data_columns: List[str]
    flow_cond_columns: List[str]
    quality_cuts: Dict
    # If None, the signal region is taken from the pre-computed 'signal_region'
    # boolean column in the raw FITS file. Set to [low, high] to derive it
    # dynamically from pm_ra values (useful when no pre-computed column exists).
    pm_ra_signal_range: Optional[Tuple[float, float]] = None
    n_extinction_iter: int = 10   # Babusiaux Gaia extinction convergence iterations


@dataclass
class TrainConfig:
    lr: float = 1e-3
    hidden_units: Union[int, List[int]] = 64
    num_layers: int = 4
    num_mc_samples: int = 10
    dropout: float = 0.0
    use_layer_norm: bool = True
    use_residual: bool = False
    activation: str = "relu"
    pct_start: float = 0.3
    noise_annealing: str = "constant"
    max_epochs: int = 100
    batch_size: int = 512
    weight_decay: float = 1e-4
    early_stopping_patience: int = 35
    seed: int = 42
    wandb_project: str = "SCREAM"
