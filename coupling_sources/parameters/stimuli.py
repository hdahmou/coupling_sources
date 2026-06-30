from __future__ import annotations

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
    def add_arguments(cls, parser: Any, prefix: str = "") -> None:
        for field in fields(cls):
            flag = "--" + "-".join(part for part in (prefix, field.name.replace("_", "-")) if part)
            if field.name == "coupled_sources":
                parser.add_argument(flag, dest=cls._dest(field.name, prefix), type=int, nargs=2, required=True)
            else:
                parser.add_argument(flag, dest=cls._dest(field.name, prefix), type=cls._arg_type(field.name), required=True)

    @classmethod
    def from_args(cls, args: Any, prefix: str = "") -> "StimuliParams":
        values = {field.name: getattr(args, cls._dest(field.name, prefix)) for field in fields(cls)}
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "StimuliParams":
        params = {field.name: values[field.name] for field in fields(cls)}
        params["coupled_sources"] = tuple(params["coupled_sources"])
        return cls(**params)

    @classmethod
    def _arg_type(cls, name: str) -> type:
        return float if name in {"sigma", "lam"} else int

    @classmethod
    def _dest(cls, name: str, prefix: str) -> str:
        return "_".join(part for part in (prefix, name) if part)

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "n_sources": self.n_sources,
            "n_rounds": self.n_rounds,
            "n_trials_full": self.n_trials_full,
            "n_trials_partial": self.n_trials_partial,
            "sigma": self.sigma,
            "lam": self.lam,
            "coupled_sources": list(self.coupled_sources),
        }

    def cli_args(self, prefix: str = "") -> list[str]:
        args: list[str] = []
        for field in fields(self):
            flag = "--" + "-".join(part for part in (prefix, field.name.replace("_", "-")) if part)
            value = getattr(self, field.name)
            if field.name == "coupled_sources":
                args.extend([flag, str(value[0]), str(value[1])])
            else:
                args.extend([flag, str(value)])
        return args
