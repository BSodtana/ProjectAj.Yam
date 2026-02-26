from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ModelConfig:
    in_dim: int = 7
    edge_dim: int = 5
    hidden_dim: int = 64
    out_dim: int = 128
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.2
    mlp_hidden_dim: int = 256
    num_classes: int = 86
    pooling: str = "mean"


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 0
    amp: bool = True
    patience: int = 5
    min_delta: float = 0.0
    device: str = "cpu"
    limit: int | None = None


@dataclass
class PathConfig:
    data_dir: str = "./data"
    output_dir: str = "./outputs"

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


@dataclass
class ProjectConfig:
    model: ModelConfig
    train: TrainConfig
    paths: PathConfig

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_project_config(
    data_dir: str = "./data",
    output_dir: str = "./outputs",
    device: str = "cpu",
    num_classes: int = 86,
) -> ProjectConfig:
    model = ModelConfig(num_classes=num_classes)
    train = TrainConfig(device=device)
    paths = PathConfig(data_dir=data_dir, output_dir=output_dir)
    return ProjectConfig(model=model, train=train, paths=paths)

