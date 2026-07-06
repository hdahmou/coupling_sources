from __future__ import annotations

import jax
import jax.numpy as jnp

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE, NO_HIDDEN_SOURCE, TRUTH_ESTIMATION
from coupling_sources.core.data import StimulusBatch, StimulusMetadata, StimulusSpec


def generate_stimulus(spec: StimulusSpec) -> StimulusBatch:
    n_trials = spec.n_trials_full + spec.n_trials_partial
    key_truth, key_error, key_hide_trial, key_hide_source = jax.random.split(
        jax.random.PRNGKey(spec.seed), 4
    )

    corr = jnp.eye(spec.n_sources)
    source_a, source_b = spec.coupled_sources
    corr = corr.at[source_a, source_b].set(spec.lam)
    corr = corr.at[source_b, source_a].set(spec.lam)
    covariance = spec.sigma**2 * corr

    truth = jax.random.uniform(
        key_truth,
        (spec.n_rounds, n_trials),
        minval=0.0,
        maxval=2 * jnp.pi,
    )
    error = (
        jax.random.normal(key_error, (spec.n_rounds, n_trials, spec.n_sources))
        @ jnp.linalg.cholesky(covariance).T
    )
    complete_reports_by_round = (truth[..., None] + error) % (2 * jnp.pi)

    hide_order = jnp.argsort(
        jax.random.uniform(key_hide_trial, (spec.n_rounds, n_trials)), axis=-1
    )
    hidden_trials = jnp.zeros((spec.n_rounds, n_trials), dtype=bool)
    hidden_trials = hidden_trials.at[
        jnp.arange(spec.n_rounds)[:, None], hide_order[:, :spec.n_trials_partial]
    ].set(True)
    hidden_index = jax.random.randint(
        key_hide_source,
        (spec.n_rounds, n_trials),
        minval=0,
        maxval=spec.n_sources,
    )
    hidden_source_by_round = jnp.where(hidden_trials, hidden_index, NO_HIDDEN_SOURCE).astype(jnp.int32)

    round_index = jnp.arange(spec.n_rounds)[:, None]
    trial_index = jnp.arange(n_trials)[None, :]
    hidden_values = jnp.where(
        hidden_trials,
        jnp.nan,
        complete_reports_by_round[round_index, trial_index, hidden_index],
    )
    visible_reports_by_round = complete_reports_by_round.at[
        round_index, trial_index, hidden_index
    ].set(hidden_values)

    trial_type_by_round = jnp.where(
        hidden_source_by_round >= 0, HIDDEN_SOURCE_PROBE, TRUTH_ESTIMATION
    ).astype(jnp.int32)
    visible_reports = visible_reports_by_round[None, ...]
    trial_type = trial_type_by_round[None, ...]
    return StimulusBatch(
        visible_reports=visible_reports,
        hidden_source=hidden_source_by_round[None, ...],
        trial_type=trial_type,
        valid_trials=jnp.ones_like(trial_type, dtype=bool),
        metadata=StimulusMetadata(
            complete_reports=complete_reports_by_round[None, ...],
            true_direction=truth[None, ...],
            source_choices=visible_reports,
            stimulus_covariance=covariance,
        ),
    )
