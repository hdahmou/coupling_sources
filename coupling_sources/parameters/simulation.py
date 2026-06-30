from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp


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

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], None]:
        return tuple(getattr(self, field.name) for field in fields(self)), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[jax.Array, ...]) -> "SimParams":
        return cls(*children)

    @classmethod
    def add_arguments(cls, parser: Any, prefix: str = "") -> None:
        for field in fields(cls):
            flag = "--" + "-".join(part for part in (prefix, field.name.replace("_", "-")) if part)
            parser.add_argument(flag, dest=cls._dest(field.name, prefix), type=float, required=True)

    @classmethod
    def from_args(cls, args: Any, prefix: str = "") -> "SimParams":
        return cls(**{field.name: getattr(args, cls._dest(field.name, prefix)) for field in fields(cls)})

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "SimParams":
        return cls(**{field.name: values[field.name] for field in fields(cls)})

    @classmethod
    def _dest(cls, name: str, prefix: str) -> str:
        return "_".join(part for part in (prefix, name) if part)

    def as_dict(self) -> dict[str, float]:
        return {field.name: float(jnp.asarray(getattr(self, field.name))) for field in fields(self)}

    def cli_args(self, prefix: str = "") -> list[str]:
        values = self.as_dict()
        args: list[str] = []
        for field in fields(self):
            flag = "--" + "-".join(part for part in (prefix, field.name.replace("_", "-")) if part)
            args.extend([flag, str(values[field.name])])
        return args
