from dataclasses import dataclass, field
import torch
from typing import List

@dataclass
class CFG:
    seed: int = 42
    data_dir: str = "./data"               
    cache_dir: str = "./cache_gnn"

    target_cols: List[str] = field(default_factory=lambda: ["Density","Tc","Tg","Rg","FFV"])
    dropout: float = 0.15
    fp16: bool = True

    hidden: int = 384
    num_layers: int = 6
    readout: str = "attn"                 

    n_folds: int = 3
    num_epochs: int = 18
    es_patience: int = 4

    batch_size: int = 192
    num_workers: int = 2

    lr: float = 1e-3
    weight_decay: float = 1e-4
    aux_lambda: float = 0.10
    edge_drop: float = 0.00

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
