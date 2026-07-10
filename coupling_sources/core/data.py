from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

Array = jax.Array | NDArray[Any]


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
    resultant_floor: float

    def __post_init__(self) -> None:
        if self.n_participants < 1:
            raise ValueError("n_participants must be at least 1")
        if self.n_sources < 1:
            raise ValueError("n_sources must be at least 1")
        if self.n_trials_full < 0 or self.n_trials_partial < 0:
            raise ValueError("trial counts must be non-negative")
        if self.n_trials_partial % self.n_sources:
            raise ValueError(
                "n_trials_partial must be a multiple of n_sources so each hidden "
                "source appears equally often"
            )

        floor_value = float(self.resultant_floor)
        if not 0.0 <= floor_value <= 1.0:
            raise ValueError("resultant_floor must be between 0 and 1")


@dataclass(frozen=True)
class StimulusMetadata:
    complete_reports: Array
    true_direction: Array
    source_choices: Array
    stimulus_covariance: Array
    resultant_length: Array

    def to_config(self) -> dict[str, Any]:
        return {
            "complete_reports": self.complete_reports,
            "true_direction": self.true_direction,
            "source_choices": self.source_choices,
            "stimulus_covariance": self.stimulus_covariance,
            "resultant_length": self.resultant_length,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> StimulusMetadata:
        return cls(**dict(config))


@dataclass(frozen=True)
class StimulusBatch:
    visible_reports: Array
    hidden_source: Array
    trial_type: Array
    valid_trials: Array
    metadata: StimulusMetadata

    def to_config(self) -> dict[str, Any]:
        return {
            "visible_reports": self.visible_reports,
            "hidden_source": self.hidden_source,
            "trial_type": self.trial_type,
            "valid_trials": self.valid_trials,
            "metadata": self.metadata.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> StimulusBatch:
        values = dict(config)
        values["metadata"] = StimulusMetadata.from_config(config["metadata"])
        return cls(**values)


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
    visible_source_mask: Array
    relative_agreement: Array
    pre_update_pair_scores: Array
    pair_scores: Array
    truth_source_scores: Array
    probe_source_scores: Array
    truth_weights: Array
    probe_weights: Array
    truth_mean_angle: Array
    hidden_mean_angle: Array
    response_mean_angle: Array

    def __getitem__(self, key: str) -> Any:
        if any(item.name == key for item in fields(self)):
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (item.name for item in fields(self))

    def __len__(self) -> int:
        return len(fields(self))

    def to_config(self) -> dict[str, Any]:
        return dict(self)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CouplingDiagnostics:
        return cls(**dict(config))


@dataclass(frozen=True)
class SimulationBatch:
    stimuli: StimulusBatch
    truth_estimates: Array
    hidden_predictions: Array
    responses: Array
    participant_parameters: Mapping[str, Any] | None = None
    diagnostics: CouplingDiagnostics | None = None

    def to_config(self) -> dict[str, Any]:
        if self.participant_parameters is None:
            raise KeyError("simulation is missing participant_parameters")
        if self.diagnostics is None:
            raise KeyError("simulation is missing diagnostics")

        return {
            "stimuli": self.stimuli.to_config(),
            "truth_estimates": self.truth_estimates,
            "hidden_predictions": self.hidden_predictions,
            "responses": self.responses,
            "participant_parameters": dict(self.participant_parameters),
            "diagnostics": self.diagnostics.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> SimulationBatch:
        values = dict(config)
        values["stimuli"] = StimulusBatch.from_config(config["stimuli"])
        values["participant_parameters"] = {
            name: float(jnp.asarray(value))
            for name, value in config["participant_parameters"].items()
        }
        values["diagnostics"] = CouplingDiagnostics.from_config(config["diagnostics"])
        return cls(**values)


def prepare_stimuli_for_fit(stimuli: StimulusBatch) -> StimulusBatch:
    """Return model-facing stimulus arrays with fit dtypes."""
    return replace(
        stimuli,
        visible_reports=jnp.asarray(stimuli.visible_reports, dtype=jnp.float64),
        hidden_source=jnp.asarray(stimuli.hidden_source, dtype=jnp.int32),
        trial_type=jnp.asarray(stimuli.trial_type, dtype=jnp.int32),
        valid_trials=jnp.asarray(stimuli.valid_trials, dtype=bool),
    )


def prepare_simulation_for_fit(simulation: SimulationBatch) -> SimulationBatch:
    """Return model-facing simulated observations with fit dtypes."""
    return replace(
        simulation,
        stimuli=prepare_stimuli_for_fit(simulation.stimuli),
        truth_estimates=jnp.asarray(simulation.truth_estimates, dtype=jnp.float64),
        hidden_predictions=jnp.asarray(
            simulation.hidden_predictions,
            dtype=jnp.float64,
        ),
    )
