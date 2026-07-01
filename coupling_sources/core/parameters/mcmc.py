from __future__ import annotations

from collections.abc import Mapping
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
    def from_mapping(cls, values: Mapping[str, Any]) -> "MCMCParams":
        return cls(**{field.name: values[field.name] for field in fields(cls)})
