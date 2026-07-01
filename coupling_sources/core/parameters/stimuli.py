from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class StimuliParams:
    seed: int
    n_sources: int
    n_rounds: int
    n_trials_full: int
    n_trials_partial: int
    sigma: float
    lam: float
    coupled_sources: tuple[int, int]

    def __post_init__(self) -> None:
        if self.n_trials_full < 0 or self.n_trials_partial < 0:
            raise ValueError("trial counts must be non-negative")

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "StimuliParams":
        params = {name: values[name] for name in cls.names()}
        params["coupled_sources"] = tuple(params["coupled_sources"])
        return cls(**params)
