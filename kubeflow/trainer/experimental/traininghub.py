from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from kubeflow_trainer_api import models

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


class TrainingHubAlgorithms(Enum):
    """Algorithm for TrainingHub Trainer."""

    SFT = "sft"
    OSFT = "osft"


@dataclass
class TrainingHubTrainer:
    func: Optional[Callable] = None
    func_args: Optional[dict] = None
    packages_to_install: Optional[list[str]] = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    env: Optional[dict[str, str]] = None
    algorithm: Optional[TrainingHubAlgorithms] = None
    # New: Optional volumes and mounts (API models are provided by caller)
    volumes: Optional[list] = None
    volume_mounts: Optional[list] = None


def get_trainer_crd_from_training_hub_trainer(
    runtime: types.Runtime,
    trainer: TrainingHubTrainer,
    initializer: Optional[types.Initializer] = None,
) -> models.TrainerV1alpha1Trainer: ...
