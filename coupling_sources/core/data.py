from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from coupling_sources.core.constants import (
    HIDDEN_SOURCE_PROBE,
    NO_HIDDEN_SOURCE,
    TRUTH_ESTIMATION,
)

Array = jax.Array | NDArray[Any]


@dataclass(frozen=True)
class GainConstants:
    g_star_y: float
    g_star_z: float

    def __post_init__(self) -> None:
        for name in ("g_star_y", "g_star_z"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_mapping(cls, values: object) -> GainConstants:
        if not isinstance(values, Mapping):
            raise ValueError("gain constants must be an object")
        return cls(
            g_star_y=float(values["g_star_y"]),
            g_star_z=float(values["g_star_z"]),
        )


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
    resultant_floor_full: float
    resultant_floor_leave_one_out: float

    def __post_init__(self) -> None:
        if self.n_participants < 1:
            raise ValueError("n_participants must be at least 1")
        if self.n_sources != 4:
            raise ValueError("n_sources must be 4 for the preregistered design")
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be at least 1")
        if self.n_trials_full < 0 or self.n_trials_partial < 0:
            raise ValueError("trial counts must be non-negative")
        if self.n_trials_partial % self.n_sources:
            raise ValueError(
                "n_trials_partial must be a multiple of n_sources so each hidden "
                "source appears equally often"
            )

        sigma = float(self.sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")

        lam = float(self.lam)
        if not math.isfinite(lam) or not 0.0 <= lam < 1.0:
            raise ValueError("lam must be finite and in [0, 1)")

        if len(self.coupled_sources) != 2:
            raise ValueError("coupled_sources must contain exactly two source indices")
        if len(set(self.coupled_sources)) != 2 or any(
            source < 0 or source >= self.n_sources for source in self.coupled_sources
        ):
            raise ValueError(
                "coupled_sources must be distinct, in-range source indices"
            )

        for name in ("resultant_floor_full", "resultant_floor_leave_one_out"):
            floor_value = float(getattr(self, name))
            if not math.isfinite(floor_value) or not 0.0 <= floor_value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")

    @classmethod
    def from_mapping(cls, values: object) -> StimulusSpec:
        if not isinstance(values, Mapping):
            raise ValueError("stimulus configuration must be an object")
        if "mean_angle_difference_band" in values:
            raise ValueError(
                "mean_angle_difference_band was removed; "
                "use the resultant floor fields"
            )
        if "resultant_floor" in values:
            raise ValueError(
                "resultant_floor was removed; use resultant_floor_full and "
                "resultant_floor_leave_one_out"
            )

        params = {item.name: values[item.name] for item in fields(cls)}
        params["coupled_sources"] = tuple(params["coupled_sources"])
        return cls(**params)


@dataclass(frozen=True)
class StimulusMetadata:
    complete_reports: Array
    true_direction: Array
    source_choices: Array
    stimulus_covariance: Array
    resultant_length_full: Array
    resultant_length_leave_one_out: Array

    def to_config(self) -> dict[str, Any]:
        return {
            "complete_reports": self.complete_reports,
            "true_direction": self.true_direction,
            "source_choices": self.source_choices,
            "stimulus_covariance": self.stimulus_covariance,
            "resultant_length_full": self.resultant_length_full,
            "resultant_length_leave_one_out": self.resultant_length_leave_one_out,
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

    def __post_init__(self) -> None:
        reports = np.asarray(self.visible_reports)
        hidden_source = np.asarray(self.hidden_source)
        trial_type = np.asarray(self.trial_type)
        valid_trials = np.asarray(self.valid_trials, dtype=bool)

        if reports.ndim != 4:
            raise ValueError(
                "visible_reports must have shape (participant, round, trial, source)"
            )
        trial_shape = reports.shape[:-1]
        for name, values in (
            ("hidden_source", hidden_source),
            ("trial_type", trial_type),
            ("valid_trials", valid_trials),
        ):
            if values.shape != trial_shape:
                raise ValueError(f"{name} must have shape {trial_shape}")

        is_truth = trial_type == TRUTH_ESTIMATION
        is_probe = trial_type == HIDDEN_SOURCE_PROBE
        if np.any(valid_trials & ~(is_truth | is_probe)):
            raise ValueError("valid trial_type values must be truth 0 or probe 1")
        if np.any(valid_trials & is_truth & (hidden_source != NO_HIDDEN_SOURCE)):
            raise ValueError("valid truth trials must use hidden_source=-1")

        n_sources = reports.shape[-1]
        probe_mask = valid_trials & is_probe
        probe_hidden = hidden_source[probe_mask]
        if np.any(
            (probe_hidden < 0)
            | (probe_hidden >= n_sources)
            | (probe_hidden != np.floor(probe_hidden))
        ):
            raise ValueError("valid probe hidden_source values must index a source")

        truth_reports = reports[valid_trials & is_truth]
        if np.any(~np.isfinite(truth_reports)):
            raise ValueError("all reports must be finite on valid truth trials")

        probe_reports = reports[probe_mask]
        if probe_reports.size:
            hidden_indices = probe_hidden.astype(np.int64, copy=False)
            hidden_is_nan = np.isnan(
                probe_reports[np.arange(probe_reports.shape[0]), hidden_indices]
            )
            if np.any(~hidden_is_nan) or np.any(
                np.sum(~np.isfinite(probe_reports), axis=-1) != 1
            ):
                raise ValueError(
                    "valid probes must contain one NaN at hidden_source and finite "
                    "reports elsewhere"
                )

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

    @classmethod
    def from_mapping(cls, values: object) -> MCMCSpecs:
        if not isinstance(values, Mapping):
            raise ValueError("MCMC configuration must be an object")
        return cls(
            seed=int(values["seed"]),
            num_warmup=int(values["num_warmup"]),
            num_samples=int(values["num_samples"]),
            num_chains=int(values["num_chains"]),
            chain_method=str(values["chain_method"]),
            target_accept_prob=float(values["target_accept_prob"]),
            max_tree_depth=int(values["max_tree_depth"]),
        )


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

    def __post_init__(self) -> None:
        trial_type = np.asarray(self.stimuli.trial_type)
        valid_trials = np.asarray(self.stimuli.valid_trials, dtype=bool)
        trial_shape = trial_type.shape
        truth_estimates = np.asarray(self.truth_estimates)
        hidden_predictions = np.asarray(self.hidden_predictions)
        responses = np.asarray(self.responses)

        for name, values in (
            ("truth_estimates", truth_estimates),
            ("hidden_predictions", hidden_predictions),
            ("responses", responses),
        ):
            if values.shape != trial_shape:
                raise ValueError(f"{name} must have shape {trial_shape}")

        truth_mask = valid_trials & (trial_type == TRUTH_ESTIMATION)
        probe_mask = valid_trials & (trial_type == HIDDEN_SOURCE_PROBE)
        if np.any(~np.isfinite(truth_estimates[truth_mask])) or np.any(
            ~np.isnan(truth_estimates[~truth_mask])
        ):
            raise ValueError(
                "truth_estimates must be finite only on valid truth trials"
            )
        if np.any(~np.isfinite(hidden_predictions[probe_mask])) or np.any(
            ~np.isnan(hidden_predictions[~probe_mask])
        ):
            raise ValueError(
                "hidden_predictions must be finite only on valid probe trials"
            )
        if np.any(~np.isfinite(responses[valid_trials])) or np.any(
            ~np.isnan(responses[~valid_trials])
        ):
            raise ValueError("responses must be finite only on valid trials")

        selected = np.where(truth_mask, truth_estimates, hidden_predictions)
        circular_difference = np.angle(
            np.exp(1j * (responses[valid_trials] - selected[valid_trials]))
        )
        if not np.allclose(circular_difference, 0.0, atol=1e-6, rtol=0.0):
            raise ValueError("responses must match the active response branch")

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
