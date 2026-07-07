from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class StimulusSpec:
    seed: int
    n_participants: int
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


def _fit_float_array(value: Any) -> jax.Array:
    return jnp.asarray(value, dtype=jnp.float64)


def prepare_stimuli_for_fit(stimuli: StimulusBatch) -> StimulusBatch:
    """Return model-facing stimulus arrays with fit dtypes."""
    return replace(
        stimuli,
        visible_reports=_fit_float_array(stimuli.visible_reports),
        hidden_source=jnp.asarray(stimuli.hidden_source, dtype=jnp.int32),
        trial_type=jnp.asarray(stimuli.trial_type, dtype=jnp.int32),
        valid_trials=jnp.asarray(stimuli.valid_trials, dtype=bool),
    )


def prepare_simulation_for_fit(simulation: SimulationBatch) -> SimulationBatch:
    """Return model-facing simulated observations with fit dtypes."""
    return replace(
        simulation,
        stimuli=prepare_stimuli_for_fit(simulation.stimuli),
        truth_estimates=_fit_float_array(simulation.truth_estimates),
        hidden_predictions=_fit_float_array(simulation.hidden_predictions),
    )
