from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class MCMCParams:
    seed: int
    num_warmup: int
    num_samples: int
    num_chains: int
    chain_method: str
    target_accept_prob: float
    max_tree_depth: int

    @classmethod
    def add_arguments(cls, parser: Any, prefix: str = "") -> None:
        for field in fields(cls):
            flag = "--" + "-".join(part for part in (prefix, field.name.replace("_", "-")) if part)
            parser.add_argument(flag, dest=cls._dest(field.name, prefix), type=cls._arg_type(field.name), required=True)

    @classmethod
    def from_args(cls, args: Any, prefix: str = "") -> "MCMCParams":
        return cls.from_mapping({field.name: getattr(args, cls._dest(field.name, prefix)) for field in fields(cls)})

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "MCMCParams":
        return cls(**{field.name: values[field.name] for field in fields(cls)})

    @classmethod
    def _arg_type(cls, name: str) -> type:
        if name in {"target_accept_prob"}:
            return float
        if name in {"chain_method"}:
            return str
        return int

    @classmethod
    def _dest(cls, name: str, prefix: str) -> str:
        return "_".join(part for part in (prefix, name) if part)

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "num_warmup": self.num_warmup,
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
            "chain_method": self.chain_method,
            "target_accept_prob": self.target_accept_prob,
            "max_tree_depth": self.max_tree_depth,
        }

    def cli_args(self, prefix: str = "") -> list[str]:
        args: list[str] = []
        values = self.as_dict()
        for field in fields(self):
            flag = "--" + "-".join(part for part in (prefix, field.name.replace("_", "-")) if part)
            args.extend([flag, str(values[field.name])])
        return args
