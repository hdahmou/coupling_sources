from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE, NO_HIDDEN_SOURCE, TRUTH_ESTIMATION
from coupling_sources.core.data import StimulusBatch, StimulusMetadata, StimulusSpec
from coupling_sources.utils.angles import mean_angle_difference, optimal_weights

_MAX_REJECTION_ATTEMPTS = 10_000


def generate_stimulus(spec: StimulusSpec) -> StimulusBatch:
    if spec.n_participants < 1:
        raise ValueError("n_participants must be at least 1")
    if spec.min_mean_angle_difference < 0.0:
        raise ValueError("min_mean_angle_difference must be non-negative")
    if spec.min_mean_angle_difference >= math.pi:
        raise ValueError("min_mean_angle_difference must be smaller than pi")

    n_trials = spec.n_trials_full + spec.n_trials_partial

    # Step 1: Build the source-noise covariance shared by participants for now.
    corr = jnp.eye(spec.n_sources)
    source_a, source_b = spec.coupled_sources
    corr = corr.at[source_a, source_b].set(spec.lam)
    corr = corr.at[source_b, source_a].set(spec.lam)
    covariance = spec.sigma**2 * corr
    error_cholesky = jnp.linalg.cholesky(covariance).T
    correlation_weights = optimal_weights(covariance).astype(covariance.dtype)
    min_mean_angle_difference = jnp.asarray(
        spec.min_mean_angle_difference,
        dtype=covariance.dtype,
    )

    def draw_source_choices(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        key_truth, key_error = jax.random.split(key, 2)
        truth = jax.random.uniform(
            key_truth,
            (n_trials,),
            minval=0.0,
            maxval=2 * jnp.pi,
            dtype=covariance.dtype,
        )
        error = (
            jax.random.normal(
                key_error,
                (n_trials, spec.n_sources),
                dtype=covariance.dtype,
            )
            @ error_cholesky
        )
        complete_reports = (truth[..., None] + error) % (2 * jnp.pi)
        return truth, complete_reports

    def draw_rejected_source_choices(
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        initial_truth = jnp.zeros((n_trials,), dtype=covariance.dtype)
        initial_reports = jnp.zeros((n_trials, spec.n_sources), dtype=covariance.dtype)
        initial_difference = jnp.full((n_trials,), -jnp.inf, dtype=covariance.dtype)
        initial_attempts = jnp.asarray(0, dtype=jnp.int32)

        def should_continue(carry: tuple[jax.Array, ...]) -> jax.Array:
            _, attempts, _, _, difference = carry
            has_rejections = jnp.any(difference < min_mean_angle_difference)
            has_attempts = attempts < _MAX_REJECTION_ATTEMPTS
            return jnp.logical_and(has_rejections, has_attempts)

        def draw_attempt(carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            key, attempts, truth, complete_reports, difference = carry
            key, sample_key = jax.random.split(key, 2)
            candidate_truth, candidate_reports = draw_source_choices(sample_key)
            candidate_difference = mean_angle_difference(
                candidate_reports,
                correlation_weights,
            )
            rejected = difference < min_mean_angle_difference
            truth = jnp.where(rejected, candidate_truth, truth)
            complete_reports = jnp.where(
                rejected[..., None],
                candidate_reports,
                complete_reports,
            )
            difference = jnp.where(rejected, candidate_difference, difference)
            return (
                key,
                attempts + 1,
                truth,
                complete_reports,
                difference,
            )

        _, _, truth, complete_reports, difference = jax.lax.while_loop(
            should_continue,
            draw_attempt,
            (
                key,
                initial_attempts,
                initial_truth,
                initial_reports,
                initial_difference,
            ),
        )
        return truth, complete_reports, difference

    # Step 2: Generate one independent round for one participant.
    def generate_round(key: jax.Array) -> tuple[jax.Array, ...]:
        key_choices, key_hide_trial, key_hide_source = jax.random.split(key, 3)
        truth, complete_reports, difference = draw_rejected_source_choices(key_choices)

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
            difference,
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
    flat_round_keys = round_keys.reshape((-1, 2))
    (
        visible_reports,
        hidden_source,
        trial_type,
        complete_reports,
        truth,
        mean_angle_differences,
    ) = jax.lax.map(generate_round, flat_round_keys)

    def unflatten_rounds(value: jax.Array) -> jax.Array:
        return value.reshape(
            (spec.n_participants, spec.n_rounds, *value.shape[1:])
        )

    visible_reports = unflatten_rounds(visible_reports)
    hidden_source = unflatten_rounds(hidden_source)
    trial_type = unflatten_rounds(trial_type)
    complete_reports = unflatten_rounds(complete_reports)
    truth = unflatten_rounds(truth)
    mean_angle_differences = unflatten_rounds(mean_angle_differences)

    if spec.min_mean_angle_difference > 0.0 and n_trials > 0:
        lowest_difference = float(jax.device_get(jnp.min(mean_angle_differences)))
        if lowest_difference < spec.min_mean_angle_difference:
            raise RuntimeError(
                "could not generate stimuli satisfying "
                "min_mean_angle_difference after "
                f"{_MAX_REJECTION_ATTEMPTS} rejection attempts; "
                f"lowest difference was {lowest_difference:.6g}"
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
