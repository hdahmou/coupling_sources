"""Reusable checks for the coupling model notebook.

The functions in this module are intentionally small and explicit. They keep
notebook cells readable while using the live model/simulator paths in
``coupling_sources.core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE, TRUTH_ESTIMATION
from coupling_sources.core.data import SimulationBatch, StimulusBatch, StimulusMetadata
from coupling_sources.core.priors import CouplingPriors
from coupling_sources.core.simulator import simulate_coupling

TWO_PI = 2.0 * np.pi
DEG = np.pi / 180.0


@dataclass(frozen=True)
class CheckDesign:
    visible_reports: jax.Array
    hidden_source: jax.Array
    trial_type: jax.Array
    valid_trials: jax.Array
    complete_reports: jax.Array
    true_direction: jax.Array
    covariance: np.ndarray
    coupled_sources: tuple[int, int]
    sigma: float
    lam: float
    diagnostic_trials: tuple[int, ...]
    probe_trials: tuple[int, ...]
    diagnostic_mask: jax.Array
    probe_mask: jax.Array
    equal_mean_angle: jax.Array
    optimal_mean_angle: jax.Array
    visible_consensus_angle: jax.Array
    partner_report_angle: jax.Array
    optimal_weights: np.ndarray
    delta_min: float

    def batch(
        self,
        *,
        include_observations: bool = False,
        simulation: SimulationBatch | None = None,
    ) -> StimulusBatch | SimulationBatch:
        stimuli = StimulusBatch(
            visible_reports=self.visible_reports,
            hidden_source=self.hidden_source,
            trial_type=self.trial_type,
            valid_trials=self.valid_trials,
            metadata=StimulusMetadata(
                complete_reports=self.complete_reports,
                true_direction=self.true_direction,
                source_choices=self.visible_reports,
                stimulus_covariance=self.covariance,
                resultant_length=jnp.ones_like(self.true_direction),
            ),
        )
        if not include_observations:
            return stimuli
        if simulation is None:
            raise ValueError("simulation is required when include_observations=True")
        return SimulationBatch(
            stimuli=stimuli,
            truth_estimates=simulation.truth_estimates,
            hidden_predictions=simulation.hidden_predictions,
            responses=simulation.responses,
        )

    def model_kwargs(
        self,
        *,
        include_observations: bool = False,
        simulation: SimulationBatch | None = None,
    ) -> dict[str, object]:
        return {
            "batch": self.batch(
                include_observations=include_observations,
                simulation=simulation,
            )
        }


def wrap_pi(angle: np.ndarray | jax.Array) -> np.ndarray:
    """Wrap radians to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % TWO_PI - np.pi


def circular_mean(angles: np.ndarray, weights: np.ndarray | None = None, axis: int = -1) -> np.ndarray:
    """Weighted circular mean with NaN-safe masking."""
    angles = np.asarray(angles, dtype=float)
    finite = np.isfinite(angles)
    clean = np.where(finite, angles, 0.0)
    if weights is None:
        weights_array = finite.astype(float)
    else:
        weights_array = np.asarray(weights, dtype=float)
        while weights_array.ndim < angles.ndim:
            weights_array = np.expand_dims(weights_array, 0)
        weights_array = np.where(finite, weights_array, 0.0)
    x = np.sum(weights_array * np.cos(clean), axis=axis)
    y = np.sum(weights_array * np.sin(clean), axis=axis)
    return np.mod(np.arctan2(y, x), TWO_PI)


def covariance_matrix(n_sources: int, sigma: float, lam: float, coupled_sources: tuple[int, int]) -> np.ndarray:
    corr = np.eye(n_sources)
    i, j = coupled_sources
    corr[i, j] = lam
    corr[j, i] = lam
    return sigma**2 * corr


def optimal_linear_weights(covariance: np.ndarray) -> np.ndarray:
    precision = np.linalg.inv(covariance)
    weights = precision @ np.ones(covariance.shape[0])
    return weights / np.sum(weights)


def benchmark_designated_weight(lam: float, n_sources: int = 4) -> float:
    """Optimal truth-estimation weight for one member of a correlated pair."""
    return 1.0 / (2.0 + (n_sources - 2.0) * (1.0 + lam))


def benchmark_weights_for_lambda(lam: float, n_sources: int = 4, coupled_sources: tuple[int, int] = (0, 1)) -> np.ndarray:
    cov = covariance_matrix(n_sources=n_sources, sigma=1.0, lam=lam, coupled_sources=coupled_sources)
    return optimal_linear_weights(cov)


def kappa_for_approx_sd_deg(sd_deg: float) -> float:
    """Large-kappa approximation: circular SD ≈ 1 / sqrt(kappa)."""
    return 1.0 / (sd_deg * DEG) ** 2


def make_locked_design(
    *,
    seed: int = 917,
    n_participants: int = 6,
    n_rounds: int = 2,
    n_trials: int = 14,
    n_sources: int = 4,
    sigma: float = 30.0 * DEG,
    lam: float = 0.8,
    coupled_sources: tuple[int, int] = (0, 1),
    probe_trials: tuple[int, ...] = (3, 7, 11),
    diagnostic_trials: tuple[int, ...] = (4, 8, 12),
    hidden_probe_source: int | tuple[int, ...] = 1,
    delta_min: float = 5.0 * DEG,
) -> CheckDesign:
    """Create a small locked design with interleaved probe and diagnostic trials.

    Trial numbers are zero-based. Probe trials hide ``hidden_probe_source``; all
    other trials are truth-estimation trials. Arrays are in the same shape
    consumed by ``model_coupling`` and ``simulate_coupling``.
    """
    rng = np.random.default_rng(seed)
    cov = covariance_matrix(n_sources=n_sources, sigma=sigma, lam=lam, coupled_sources=coupled_sources)
    true_direction = rng.uniform(0.0, TWO_PI, size=(n_participants, n_rounds, n_trials))
    errors = rng.multivariate_normal(np.zeros(n_sources), cov, size=(n_participants, n_rounds, n_trials))
    complete_reports = np.mod(true_direction[..., None] + errors, TWO_PI)
    visible_reports = complete_reports.copy()
    hidden_source = -np.ones((n_participants, n_rounds, n_trials), dtype=np.int32)
    trial_type = np.zeros((n_participants, n_rounds, n_trials), dtype=np.int32)

    hidden_sequence = (hidden_probe_source,) if isinstance(hidden_probe_source, int) else tuple(hidden_probe_source)
    if len(hidden_sequence) == 0:
        raise ValueError("hidden_probe_source cannot be empty")
    for probe_index, trial in enumerate(probe_trials):
        hidden_index = int(hidden_sequence[probe_index % len(hidden_sequence)])
        hidden_source[:, :, trial] = hidden_index
        trial_type[:, :, trial] = HIDDEN_SOURCE_PROBE
        visible_reports[:, :, trial, hidden_index] = np.nan

    valid_trials = np.ones((n_participants, n_rounds, n_trials), dtype=bool)
    diagnostic_mask = np.zeros((n_participants, n_rounds, n_trials), dtype=bool)
    for trial in diagnostic_trials:
        diagnostic_mask[:, :, trial] = True
    diagnostic_mask &= trial_type == TRUTH_ESTIMATION
    probe_mask = trial_type == HIDDEN_SOURCE_PROBE

    opt_weights = optimal_linear_weights(cov)
    equal_mean = circular_mean(complete_reports, axis=-1)
    optimal_mean = circular_mean(complete_reports, weights=opt_weights, axis=-1)
    visible_consensus = circular_mean(visible_reports, axis=-1)

    i, j = coupled_sources
    partner_for_hidden = np.full_like(hidden_source, i)
    partner_for_hidden = np.where(hidden_source == i, j, partner_for_hidden)
    partner_for_hidden = np.where(hidden_source == j, i, partner_for_hidden)
    partner_report = np.take_along_axis(complete_reports, partner_for_hidden[..., None], axis=-1)[..., 0]
    partner_report = np.where(probe_mask, partner_report, np.nan)

    anchor_gap = np.abs(wrap_pi(optimal_mean - equal_mean))
    diagnostic_mask &= anchor_gap >= delta_min

    return CheckDesign(
        visible_reports=jnp.asarray(visible_reports),
        hidden_source=jnp.asarray(hidden_source),
        trial_type=jnp.asarray(trial_type),
        valid_trials=jnp.asarray(valid_trials),
        complete_reports=jnp.asarray(complete_reports),
        true_direction=jnp.asarray(true_direction),
        covariance=cov,
        coupled_sources=coupled_sources,
        sigma=sigma,
        lam=lam,
        diagnostic_trials=diagnostic_trials,
        probe_trials=probe_trials,
        diagnostic_mask=jnp.asarray(diagnostic_mask),
        probe_mask=jnp.asarray(probe_mask),
        equal_mean_angle=jnp.asarray(equal_mean),
        optimal_mean_angle=jnp.asarray(optimal_mean),
        visible_consensus_angle=jnp.asarray(visible_consensus),
        partner_report_angle=jnp.asarray(partner_report),
        optimal_weights=opt_weights,
        delta_min=delta_min,
    )


def sim_params_from_prior_draw(prior_draws: Mapping[str, np.ndarray | jax.Array], index: int) -> dict[str, float]:
    draw_count = len(np.asarray(prior_draws["α"]).reshape(-1))
    c0 = np.asarray(prior_draws.get("c0", np.zeros(draw_count))).reshape(-1)[index]
    return {
        "α": float(np.asarray(prior_draws["α"]).reshape(-1)[index]),
        "g_y": float(np.asarray(prior_draws["g_y"]).reshape(-1)[index]),
        "g_z": float(np.asarray(prior_draws["g_z"]).reshape(-1)[index]),
        "κ": float(np.asarray(prior_draws["κ"]).reshape(-1)[index]),
        "c0": float(c0),
    }


def simulate_design(seed: int, design: CheckDesign, params: Mapping[str, float]) -> SimulationBatch:
    return simulate_coupling(
        seed=seed,
        batch=design.batch(),
        parameters=params,
    )


def simulate_prior_dataset(seed: int, design: CheckDesign, prior_draws: Mapping[str, np.ndarray | jax.Array], index: int) -> SimulationBatch:
    return simulate_design(seed, design, sim_params_from_prior_draw(prior_draws, index))


def score_contrast(states, coupled_sources: tuple[int, int] = (0, 1)) -> np.ndarray:
    scores = np.asarray(states.pair_scores)
    n_sources = scores.shape[-1]
    i, j = coupled_sources
    designated = 0.5 * (scores[..., i, j] + scores[..., j, i])
    off_diag = ~np.eye(n_sources, dtype=bool)
    other_mask = off_diag.copy()
    other_mask[i, j] = False
    other_mask[j, i] = False
    other = scores[..., other_mask].mean(axis=-1)
    return designated - other


def probe_freezing_delta(states, hidden_source: jax.Array) -> float:
    before = np.asarray(states.pre_update_pair_scores)
    after = np.asarray(states.pair_scores)
    hidden = np.asarray(hidden_source)
    probe_locations = np.argwhere(hidden >= 0)
    if len(probe_locations) == 0:
        return 0.0
    max_delta = 0.0
    for p, r, t in probe_locations:
        h = int(hidden[p, r, t])
        mask = np.zeros(after.shape[-2:], dtype=bool)
        mask[h, :] = True
        mask[:, h] = True
        delta = np.max(np.abs(after[p, r, t][mask] - before[p, r, t][mask]))
        max_delta = max(max_delta, float(delta))
    return max_delta


def correction_index_values(responses: np.ndarray, design: CheckDesign) -> np.ndarray:
    response = np.asarray(responses, dtype=float)
    equal = np.asarray(design.equal_mean_angle)
    optimal = np.asarray(design.optimal_mean_angle)
    gap = wrap_pi(optimal - equal)
    movement = wrap_pi(response - equal)
    ci = movement / np.where(np.abs(gap) < design.delta_min, np.nan, gap)
    return np.where(np.asarray(design.diagnostic_mask), ci, np.nan)


def partner_proximity_values(responses: np.ndarray, design: CheckDesign) -> np.ndarray:
    response = np.asarray(responses, dtype=float)
    consensus = np.asarray(design.visible_consensus_angle)
    partner = np.asarray(design.partner_report_angle)
    gap = wrap_pi(partner - consensus)
    movement = wrap_pi(response - consensus)
    pp = movement / np.where(np.abs(gap) < design.delta_min, np.nan, gap)
    return np.where(np.asarray(design.probe_mask), pp, np.nan)


def dataset_indices(simulation: SimulationBatch, design: CheckDesign) -> dict[str, float]:
    ci = correction_index_values(np.asarray(simulation.truth_estimates), design)
    pp = partner_proximity_values(np.asarray(simulation.hidden_predictions), design)
    return {"CI": float(np.nanmean(ci)), "PP": float(np.nanmean(pp))}


def angular_error_values(simulation: SimulationBatch, design: CheckDesign) -> dict[str, np.ndarray]:
    truth_error = wrap_pi(np.asarray(simulation.truth_estimates) - np.asarray(design.equal_mean_angle))
    probe_error = wrap_pi(np.asarray(simulation.hidden_predictions) - np.asarray(design.visible_consensus_angle))
    return {
        "truth": truth_error[np.isfinite(truth_error)],
        "probe": probe_error[np.isfinite(probe_error)],
    }


def parameter_transforms(samples: Mapping[str, np.ndarray | jax.Array]) -> dict[str, np.ndarray]:
    alpha = np.asarray(samples["α"], dtype=float)
    g_y = np.asarray(samples["g_y"], dtype=float)
    g_z = np.asarray(samples["g_z"], dtype=float)
    eps = np.finfo(float).eps
    scaled_g_y = np.clip(g_y / 10.0, eps, 1 - eps)
    scaled_g_z = np.clip(g_z / 10.0, eps, 1 - eps)
    return {
        "logit_α": np.log(np.clip(alpha, eps, 1 - eps) / np.clip(1 - alpha, eps, 1)),
        "log_g_y": np.log(g_y),
        "log_g_z": np.log(g_z),
        "logit_g_y_over_10": np.log(scaled_g_y / (1 - scaled_g_y)),
        "logit_g_z_over_10": np.log(scaled_g_z / (1 - scaled_g_z)),
        "log_κ": np.log(np.asarray(samples["κ"], dtype=float)),
    }


def posterior_draw_dict(samples: Mapping[str, np.ndarray | jax.Array], draw_index: int) -> dict[str, float]:
    return {
        name: float(np.asarray(samples[name]).reshape(-1)[draw_index])
        for name in CouplingPriors.parameter_sites()
    }


def states_for_params(design: CheckDesign, params: Mapping[str, float]) -> object:
    simulation = simulate_design(0, design, params)
    return SimpleNamespace(**simulation.diagnostics)


def response_log_likelihood_by_trial(simulation: SimulationBatch, design: CheckDesign, params: Mapping[str, float]) -> np.ndarray:
    states = states_for_params(design, params)
    kappa = params["κ"]
    truth_logp = np.asarray(dist.VonMises(states.truth_mean_angle, kappa).log_prob(simulation.truth_estimates))
    probe_logp = np.asarray(dist.VonMises(states.hidden_mean_angle, kappa).log_prob(simulation.hidden_predictions))
    trial_type = np.asarray(design.trial_type)
    valid = np.asarray(design.valid_trials)
    logp = np.where(trial_type == HIDDEN_SOURCE_PROBE, probe_logp, truth_logp)
    return np.where(valid, logp, np.nan)


def posterior_predictive(
    *,
    seed: int,
    samples: Mapping[str, np.ndarray | jax.Array],
    design: CheckDesign,
    max_draws: int = 100,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    flat_n = len(np.asarray(samples["α"]).reshape(-1))
    draw_indices = np.linspace(0, flat_n - 1, min(max_draws, flat_n), dtype=int)
    responses = []
    means = []
    kappas = []
    truth_means = []
    hidden_means = []
    for draw_index in draw_indices:
        params = posterior_draw_dict(samples, int(draw_index))
        states = states_for_params(design, params)
        truth_mean = np.asarray(states.truth_mean_angle)
        hidden_mean = np.asarray(states.hidden_mean_angle)
        trial_type = np.asarray(design.trial_type)
        mean = np.where(trial_type == HIDDEN_SOURCE_PROBE, hidden_mean, truth_mean)
        replicated = rng.vonmises(mean, params["κ"]) % TWO_PI
        replicated = np.where(np.asarray(design.valid_trials), replicated, np.nan)
        responses.append(replicated)
        means.append(mean)
        kappas.append(params["κ"])
        truth_means.append(truth_mean)
        hidden_means.append(hidden_mean)
    return {
        "draw_indices": np.asarray(draw_indices),
        "responses": np.asarray(responses),
        "response_mean_angle": np.asarray(means),
        "truth_mean_angle": np.asarray(truth_means),
        "hidden_mean_angle": np.asarray(hidden_means),
        "κ": np.asarray(kappas),
    }


def flatten_valid(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(values)
    if valid is None:
        return array[np.isfinite(array)]
    mask = np.asarray(valid) & np.isfinite(array)
    return array[mask]
