from __future__ import annotations

import jax
import jax.numpy as jnp

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE, NO_HIDDEN_SOURCE, TRUTH_ESTIMATION
from coupling_sources.core.data import StimulusBatch, StimulusMetadata, StimulusSpec


def generate_stimulus(spec: StimulusSpec) -> StimulusBatch:
    if spec.n_participants < 1:
        raise ValueError("n_participants must be at least 1")

    n_trials = spec.n_trials_full + spec.n_trials_partial

    # Step 1: Build the source-noise covariance shared by participants for now.
    corr = jnp.eye(spec.n_sources)
    source_a, source_b = spec.coupled_sources
    corr = corr.at[source_a, source_b].set(spec.lam)
    corr = corr.at[source_b, source_a].set(spec.lam)
    covariance = spec.sigma**2 * corr
    error_cholesky = jnp.linalg.cholesky(covariance).T

    # Step 2: Generate one independent round for one participant.
    def generate_round(
        key: jax.Array,
        participant_error_cholesky: jax.Array,
    ) -> tuple[jax.Array, ...]:
        key_truth, key_error, key_hide_trial, key_hide_source = jax.random.split(key, 4)

        truth = jax.random.uniform(
            key_truth,
            (n_trials,),
            minval=0.0,
            maxval=2 * jnp.pi,
        )
        error = (
            jax.random.normal(key_error, (n_trials, spec.n_sources))
            @ participant_error_cholesky
        )
        complete_reports = (truth[..., None] + error) % (2 * jnp.pi)

        hide_order = jnp.argsort(
            jax.random.uniform(key_hide_trial, (n_trials,)),
            axis=-1,
        )
        hidden_trials = jnp.zeros((n_trials,), dtype=bool)
        hidden_trials = hidden_trials.at[hide_order[:spec.n_trials_partial]].set(True)
        hidden_index = jax.random.randint(
            key_hide_source,
            (n_trials,),
            minval=0,
            maxval=spec.n_sources,
        )
        hidden_source = jnp.where(
            hidden_trials,
            hidden_index,
            NO_HIDDEN_SOURCE,
        ).astype(jnp.int32)

        trial_index = jnp.arange(n_trials)
        hidden_values = jnp.where(
            hidden_trials,
            jnp.nan,
            complete_reports[trial_index, hidden_index],
        )
        visible_reports = complete_reports.at[trial_index, hidden_index].set(
            hidden_values
        )

        trial_type = jnp.where(
            hidden_source >= 0, HIDDEN_SOURCE_PROBE, TRUTH_ESTIMATION
        ).astype(jnp.int32)
        return (
            visible_reports,
            hidden_source,
            trial_type,
            complete_reports,
            truth,
        )

    # Step 3: Derive a deterministic PRNG key for each participant and round.
    base_key = jax.random.PRNGKey(spec.seed)
    round_indices = jnp.arange(spec.n_rounds)

    def round_keys_for_participant(participant: jax.Array) -> jax.Array:
        participant_key = jax.random.fold_in(base_key, participant)
        return jax.vmap(
            lambda round_index: jax.random.fold_in(participant_key, round_index)
        )(round_indices)

    round_keys = jax.vmap(round_keys_for_participant)(jnp.arange(spec.n_participants))

    # Step 4: Map one-round generation over rounds, then participants.
    generate_rounds = jax.vmap(generate_round, in_axes=(0, None))
    (
        visible_reports,
        hidden_source,
        trial_type,
        complete_reports,
        truth,
    ) = jax.vmap(generate_rounds, in_axes=(0, None))(
        round_keys,
        error_cholesky,
    )

    # Step 5: Package arrays in the model's participant, round, trial, source shape.
    return StimulusBatch(
        visible_reports=visible_reports,
        hidden_source=hidden_source,
        trial_type=trial_type,
        valid_trials=jnp.ones_like(trial_type, dtype=bool),
        metadata=StimulusMetadata(
            complete_reports=complete_reports,
            true_direction=truth,
            source_choices=visible_reports,
            stimulus_covariance=covariance,
        ),
    )
