from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist


Scalar = float | jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SimParams:
    alpha: Scalar
    gamma: Scalar
    kappa_p: Scalar
    kappa_a: Scalar

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, jnp.asarray(getattr(self, field.name)))

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], None]:
        return tuple(getattr(self, name) for name in self.names()), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[jax.Array, ...]) -> "SimParams":
        return cls(*children)

    @staticmethod
    def prior_distributions() -> dict[str, dist.Distribution]:
        return {
            "alpha": dist.Beta(2.0, 2.0),
            "gamma": dist.Beta(2.0, 2.0),
            "kappa_p": dist.Gamma(concentration=5.0, rate=1.0 / 3.0),
            "kappa_a": dist.Gamma(concentration=5.0, rate=1.0 / 3.0),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SimParams":
        return cls(**{name: values[name] for name in cls.names()})

    def as_dict(self) -> dict[str, float]:
        return {name: float(jnp.asarray(getattr(self, name))) for name in self.names()}
