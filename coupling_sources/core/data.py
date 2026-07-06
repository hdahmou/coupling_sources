from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import jax


@dataclass(frozen=True)
class StimulusSpec:
    seed: int
    n_sources: int
    n_rounds: int
    n_trials_full: int
    n_trials_partial: int
    sigma: float
    lam: float
    coupled_sources: tuple[int, int]


@dataclass(frozen=True)
class StimulusMetadata:
    complete_reports: jax.Array
    true_direction: jax.Array
    source_choices: jax.Array
    stimulus_covariance: jax.Array


@dataclass(frozen=True)
class StimulusBatch:
    visible_reports: jax.Array
    hidden_source: jax.Array
    trial_type: jax.Array
    valid_trials: jax.Array
    metadata: StimulusMetadata


@dataclass(frozen=True)
class MCMCSpecs:
    seed: int
    num_warmup: int
    num_samples: int
    num_chains: int
    chain_method: str
    target_accept_prob: float
    max_tree_depth: int


@dataclass(frozen=True)
class CouplingDiagnostics(Mapping[str, Any]):
    visible_source_mask: jax.Array
    relative_agreement: jax.Array
    pre_update_pair_scores: jax.Array
    pair_scores: jax.Array
    truth_source_scores: jax.Array
    probe_source_scores: jax.Array
    truth_weights: jax.Array
    probe_weights: jax.Array
    truth_mean_angle: jax.Array
    hidden_mean_angle: jax.Array
    response_mean_angle: jax.Array

    def __getitem__(self, key: str) -> Any:
        if any(item.name == key for item in fields(self)):
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (item.name for item in fields(self))

    def __len__(self) -> int:
        return len(fields(self))


@dataclass(frozen=True)
class SimulationBatch:
    stimuli: StimulusBatch
    truth_estimates: jax.Array
    hidden_predictions: jax.Array
    responses: jax.Array
    participant_parameters: Mapping[str, Any] | None = None
    diagnostics: CouplingDiagnostics | Mapping[str, Any] = field(default_factory=dict)

    @property
    def visible_reports(self) -> jax.Array:
        return self.stimuli.visible_reports

    @property
    def hidden_source(self) -> jax.Array:
        return self.stimuli.hidden_source

    @property
    def trial_type(self) -> jax.Array:
        return self.stimuli.trial_type

    @property
    def valid_trials(self) -> jax.Array:
        return self.stimuli.valid_trials

    @property
    def metadata(self) -> StimulusMetadata:
        return self.stimuli.metadata
