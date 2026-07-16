"""Direct response simulator for model-shaped coupling stimuli."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE, TRUTH_ESTIMATION
from coupling_sources.core.data import (
    CouplingDiagnostics,
    GainConstants,
    SimulationBatch,
    StimulusBatch,
)
from coupling_sources.core.priors import CouplingPriors
from coupling_sources.utils.angles import circular_mean


def simulate_coupling(
    *,
    seed: int,
    batch: StimulusBatch,
    parameters: Mapping[str, Any],
    gains: GainConstants,
) -> SimulationBatch:
    """Simulate responses using parameters that include the fixed ``c0``."""
    params = CouplingPriors.normalize_parameters(parameters)
    params["g_y"] = params["gamma_y"] * gains.g_star_y
    params["g_z"] = params["gamma_z"] * gains.g_star_z
    response_key = jax.random.PRNGKey(seed)

    visible_reports = jnp.asarray(batch.visible_reports, dtype=jnp.float64)
    hidden_source = jnp.asarray(batch.hidden_source, dtype=jnp.int32)
    trial_type = jnp.asarray(batch.trial_type, dtype=jnp.int32)
    valid_trials = jnp.asarray(batch.valid_trials, dtype=bool)

    n_participants, n_rounds, n_trials, n_sources = visible_reports.shape
    off_diagonal_bool = ~jnp.eye(n_sources, dtype=bool)
    pair_scores = jnp.broadcast_to(
        (1.0 - jnp.eye(n_sources)) * params["c0"],
        (n_participants, n_rounds, n_sources, n_sources),
    )

    trial_shape = (n_participants, n_rounds, n_trials)
    source_shape = (*trial_shape, n_sources)
    pair_shape = (*source_shape, n_sources)
    diagnostics = {
        "visible_source_mask": jnp.zeros(source_shape, dtype=bool),
        "relative_agreement": jnp.zeros(pair_shape, dtype=pair_scores.dtype),
        "pre_update_pair_scores": jnp.zeros(pair_shape, dtype=pair_scores.dtype),
        "pair_scores": jnp.zeros(pair_shape, dtype=pair_scores.dtype),
        "truth_source_scores": jnp.zeros(source_shape, dtype=pair_scores.dtype),
        "probe_source_scores": jnp.zeros(source_shape, dtype=pair_scores.dtype),
        "truth_weights": jnp.zeros(source_shape, dtype=visible_reports.dtype),
        "probe_weights": jnp.zeros(source_shape, dtype=visible_reports.dtype),
        "truth_mean_angle": jnp.zeros(trial_shape, dtype=visible_reports.dtype),
        "hidden_mean_angle": jnp.zeros(trial_shape, dtype=visible_reports.dtype),
        "response_mean_angle": jnp.zeros(trial_shape, dtype=visible_reports.dtype),
    }

    for trial_index in range(n_trials):
        reports_raw = visible_reports[:, :, trial_index, :]
        visible_sources = jnp.isfinite(reports_raw)
        reports = jnp.nan_to_num(reports_raw, nan=0.0)

        visible_pairs = (
            visible_sources[..., :, None]
            & visible_sources[..., None, :]
            & off_diagonal_bool
        )
        pair_agreement = jnp.cos(reports[..., :, None] - reports[..., None, :])
        visible_pair_values = pair_agreement * visible_pairs.astype(pair_agreement.dtype)
        pair_count = jnp.maximum(jnp.sum(visible_pairs, axis=(-2, -1), keepdims=True), 1)
        mean_visible_agreement = jnp.sum(visible_pair_values, axis=(-2, -1), keepdims=True) / pair_count
        relative_agreement = 0.5 * (pair_agreement - mean_visible_agreement) * visible_pairs.astype(pair_agreement.dtype)

        pre_update_pair_scores = pair_scores
        valid_trial = valid_trials[:, :, trial_index]
        update_pairs = (visible_pairs & valid_trial[..., None, None]).astype(pair_scores.dtype)
        pair_scores = pair_scores + update_pairs * params["learningRate"] * (relative_agreement - pair_scores)

        visible_count = jnp.maximum(jnp.sum(visible_sources, axis=-1, keepdims=True), 1)
        truth_denominator = jnp.maximum(visible_count - 1, 1)
        other_visible_sources = visible_sources[..., None, :] & off_diagonal_bool
        truth_source_scores = -jnp.sum(
            pair_scores * other_visible_sources.astype(pair_scores.dtype),
            axis=-1,
        ) / truth_denominator
        truth_source_scores = truth_source_scores * visible_sources.astype(pair_scores.dtype)

        hidden = hidden_source[:, :, trial_index]
        hidden_indices = jnp.broadcast_to(
            jnp.clip(hidden, 0, n_sources - 1)[..., None, None],
            pair_scores.shape[:-1] + (1,),
        )
        probe_source_scores = jnp.take_along_axis(pair_scores, hidden_indices, axis=-1)[..., 0]
        probe_source_scores = probe_source_scores * visible_sources.astype(pair_scores.dtype)

        no_visible_sources = ~jnp.any(visible_sources, axis=-1, keepdims=True)
        fallback_source = jnp.arange(n_sources) == 0
        safe_visible_sources = visible_sources | (
            no_visible_sources & fallback_source
        )
        log_visible_sources = jnp.log(safe_visible_sources)
        truth_weights = jax.nn.softmax(truth_source_scores * params["g_y"] + log_visible_sources, axis=-1)
        probe_weights = jax.nn.softmax(probe_source_scores * params["g_z"] + log_visible_sources, axis=-1)
        truth_mean_angle = circular_mean(reports, truth_weights)
        hidden_mean_angle = circular_mean(reports, probe_weights)
        response_mean_angle = jnp.where(
            trial_type[:, :, trial_index] == HIDDEN_SOURCE_PROBE,
            hidden_mean_angle,
            truth_mean_angle,
        )

        trial_values = {
            "visible_source_mask": visible_sources,
            "relative_agreement": relative_agreement,
            "pre_update_pair_scores": pre_update_pair_scores,
            "pair_scores": pair_scores,
            "truth_source_scores": truth_source_scores,
            "probe_source_scores": probe_source_scores,
            "truth_weights": truth_weights,
            "probe_weights": probe_weights,
            "truth_mean_angle": truth_mean_angle,
            "hidden_mean_angle": hidden_mean_angle,
            "response_mean_angle": response_mean_angle,
        }
        for name, value in trial_values.items():
            diagnostics[name] = diagnostics[name].at[:, :, trial_index, ...].set(value)

    key_truth, key_hidden = jax.random.split(response_key)
    truth_draws = dist.VonMises(diagnostics["truth_mean_angle"], params["vmConcentration"]).sample(key_truth) % (2 * jnp.pi)
    hidden_draws = dist.VonMises(diagnostics["hidden_mean_angle"], params["vmConcentration"]).sample(key_hidden) % (2 * jnp.pi)
    is_truth = (trial_type == TRUTH_ESTIMATION) & valid_trials
    is_probe = (trial_type == HIDDEN_SOURCE_PROBE) & valid_trials
    truth_estimates = jnp.where(is_truth, truth_draws, jnp.nan)
    hidden_predictions = jnp.where(is_probe, hidden_draws, jnp.nan)
    responses = jnp.where(valid_trials, jnp.where(is_probe, hidden_draws, truth_draws), jnp.nan)

    return SimulationBatch(
        stimuli=StimulusBatch(
            visible_reports=visible_reports,
            hidden_source=hidden_source,
            trial_type=trial_type,
            valid_trials=valid_trials,
            metadata=batch.metadata,
        ),
        truth_estimates=truth_estimates,
        hidden_predictions=hidden_predictions,
        responses=responses,
        participant_parameters=params,
        diagnostics=CouplingDiagnostics(**diagnostics),
    )
