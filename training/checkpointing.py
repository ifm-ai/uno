# Portions adapted from LlamaFactory, Copyright 2025 the LlamaFactory team,
# under the Apache License 2.0.

from __future__ import annotations

from functools import WRAPPER_ASSIGNMENTS, partial, wraps
from types import MethodType

import torch


def _checkpoint_trainable_layers(checkpoint_function):
    """Checkpoint only modules with trainable weights and seed their input grad."""

    @wraps(checkpoint_function, assigned=WRAPPER_ASSIGNMENTS + ("__self__",))
    def wrapped(function, *args, **kwargs):
        module = function.func.__self__ if isinstance(function, partial) else function.__self__
        if any(parameter.requires_grad for parameter in module.parameters()):
            for argument in args:
                if torch.is_tensor(argument) and torch.is_floating_point(argument):
                    argument.requires_grad_(True)
                    break
            return checkpoint_function(function, *args, **kwargs)
        return function(*args, **kwargs)

    return wrapped


def _enable_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
    from torch.utils.checkpoint import checkpoint

    if not self.supports_gradient_checkpointing:
        raise ValueError(f"{self.__class__.__name__} does not support checkpointing.")
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}
    checkpoint_function = _checkpoint_trainable_layers(
        partial(checkpoint, **gradient_checkpointing_kwargs)
    )
    self._set_gradient_checkpointing(
        enable=True, gradient_checkpointing_func=checkpoint_function
    )


def configure_gradient_checkpointing(model) -> None:
    """Install the proven LoRA-aware checkpointing behavior before PEFT wrapping."""
    model.gradient_checkpointing_enable = MethodType(
        _enable_gradient_checkpointing, model
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )
    model.config.use_cache = False
