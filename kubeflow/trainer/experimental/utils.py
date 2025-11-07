from typing import NoReturn, Optional

from kubeflow_trainer_api import models

from kubeflow.trainer.experimental import (
    ExperimentalTrainer,
    traininghub,
    transformers,
)
from kubeflow.trainer.types import types


def get_trainer_crd_from_experimental_trainer(
    runtime: types.Runtime,
    trainer: ExperimentalTrainer,
    initializer: Optional[types.Initializer] = None,
) -> models.TrainerV1alpha1Trainer:
    if isinstance(trainer, traininghub.TrainingHubTrainer):
        return traininghub.get_trainer_crd_from_training_hub_trainer(
            runtime,
            trainer,
            initializer,
        )

    elif isinstance(trainer, transformers.TransformersTrainer):
        return transformers.get_trainer_crd_from_transformers_trainer(
            runtime,
            trainer,
            initializer,
        )

    else:
        # This is an unknown trainer. The assertion is done in a function so we can use
        # a type checker to verify that trainer is a "Never" type.
        _raise_unknown_experimental_trainer(trainer)


def _raise_unknown_experimental_trainer(trainer: NoReturn) -> NoReturn:
    raise ValueError(f"Unknown trainer {trainer}.")
