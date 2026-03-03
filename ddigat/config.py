from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ModelConfig:
    in_dim: int = 7
    edge_dim: int = 5
    encoder_type: str = "gat"
    hidden_dim: int = 64
    out_dim: int = 128
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.2
    mlp_hidden_dim: int = 256
    num_classes: int = 86
    pooling: str = "mean"
    use_ecfp_features: bool = False
    use_physchem_features: bool = False
    use_maccs_features: bool = False
    ecfp_bits: int = 2048
    ecfp_radius: int = 2
    physchem_dim: int = 0
    maccs_dim: int = 166
    ecfp_proj_dim: int = 128
    physchem_proj_dim: int = 32
    maccs_proj_dim: int = 32


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
    use_class_weights: bool = False
    class_weight_method: str = "inv_sqrt"
    class_weight_normalize: str = "sample_mean"
    class_weight_beta: float = 0.9999
    class_weight_clip_min: float = 0.25
    class_weight_clip_max: float = 4.0
    class_weight_eps: float = 1e-12
    class_counts: list[int] | None = None
    label_smoothing: float = 0.0
    logit_adjust_tau: float = 0.0
    logit_adjust_eps: float = 1e-12
    split_strategy: str = "cold_drug"
    split_seed: int = 42
    cold_k: int = 5
    cold_fold: int = 0
    cold_protocol: str = "s1"
    cold_min_test_pairs: int = 5000
    cold_min_test_labels: int = 45
    cold_max_resamples: int = 200
    cold_dedupe_policy: str = "keep_all"
    cold_write_legacy_flat_splits: bool = False
    training_start_unix: float | None = None


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
