from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class StreamConfig:
    name: str
    raw_data_path: str
    generated_data_path: str
    features: List[str]
    error_features: List[str]
    pm_ra_signal_range: Tuple[float, float]
    quality_cuts: Dict


@dataclass
class TrainConfig:
    lr: float = 1e-3
    hidden_units: int = 64
    num_layers: int = 4
    num_mc_samples: int = 10
    dropout: float = 0.0
    use_layer_norm: bool = True
    use_residual: bool = False
    # Placeholder — noise annealing will be removed in a later pass when full
    # covariance matrix inputs are implemented.
    noise_annealing: str = "constant"
    max_epochs: int = 100
    batch_size: int = 512
    weight_decay: float = 1e-4
    seed: int = 42
    wandb_project: str = "SCREAM"
