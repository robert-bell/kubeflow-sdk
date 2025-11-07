from typing import Union

from kubeflow.trainer.experimental.traininghub import TrainingHubAlgorithms, TrainingHubTrainer
from kubeflow.trainer.experimental.transformers import TransformersTrainer

__all__ = (
    "ExperimentalTrainer",
    "TrainingHubAlgorithms",
    "TrainingHubTrainer",
    "TransformersTrainer",
)

ExperimentalTrainer = Union[TransformersTrainer, TrainingHubTrainer]
