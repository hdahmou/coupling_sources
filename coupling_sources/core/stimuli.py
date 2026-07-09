from __future__ import annotations

import jax
import jax.numpy as jnp

from coupling_sources.core.constants import HIDDEN_SOURCE_PROBE, NO_HIDDEN_SOURCE, TRUTH_ESTIMATION
from coupling_sources.core.data import StimulusBatch, StimulusMetadata, StimulusSpec

_MAX_REJECTION_ATTEMPTS = 10_000


def generate_stimulus(spec: StimulusSpec) -> StimulusBatch:
    n_trials = spec.n_trials_full + spec.n_trials_partial

    corr = jnp.eye(spec.n_sources)
    a, b = spec.coupled_sources
    corr = corr.at[a, b].set(spec.lam).at[b, a].set(spec.lam)
    covariance = spec.sigma**2 * corr
    error_cholesky = jnp.linalg.cholesky(covariance).T
    resultant_floor = jnp.asarray(spec.resultant_floor, dtype=covariance.dtype)

    base_key = jax.random.PRNGKey(spec.seed)
    rounds = jnp.arange(spec.n_rounds)

    def keys_for_participant(participant: jax.Array) -> jax.Array:
        participant_key = jax.random.fold_in(base_key, participant)
        return jax.vmap(lambda round_: jax.random.fold_in(participant_key, round_))(
            rounds
        )

    round_keys = jax.vmap(keys_for_participant)(jnp.arange(spec.n_participants))

    def generate_round(key: jax.Array) -> tuple[jax.Array, ...]:
        key_schedule, key_trials = jax.random.split(key)
        trial_type, hidden_source = _draw_trial_schedule(key_schedule, spec, n_trials)
        truth, complete_reports, resultant_length = _draw_until_resultant_floor(
            key_trials,
            spec,
            n_trials,
            error_cholesky,
            resultant_floor,
        )

        is_probe = trial_type == HIDDEN_SOURCE_PROBE
        hide_index = jnp.clip(hidden_source, 0, spec.n_sources - 1)
        trials = jnp.arange(n_trials)
        visible_reports = complete_reports.at[trials, hide_index].set(
            jnp.where(is_probe, jnp.nan, complete_reports[trials, hide_index])
        )

        return (
            visible_reports,
            hidden_source,
            trial_type,
            complete_reports,
            truth,
            resultant_length,
        )

    flat_rounds = jax.lax.map(
        generate_round,
        round_keys.reshape((-1, 2)),
    )
    round_grids = tuple(
        value.reshape((spec.n_participants, spec.n_rounds, *value.shape[1:]))
        for value in flat_rounds
    )
    resultant_length = round_grids[-1]

    if n_trials > 0 and spec.resultant_floor > 0.0:
        lowest = float(jax.device_get(jnp.min(resultant_length)))
        if lowest < spec.resultant_floor:
            raise RuntimeError(
                f"could not satisfy resultant_floor after {_MAX_REJECTION_ATTEMPTS} "
                f"attempts; lowest resultant was {lowest:.6g}, "
                f"floor was {spec.resultant_floor:.6g}"
            )

    (
        visible_reports,
        hidden_source,
        trial_type,
        complete_reports,
        truth,
        resultant_length,
    ) = round_grids

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
            resultant_length=resultant_length,
        ),
    )


def _draw_trial_schedule(
    key: jax.Array,
    spec: StimulusSpec,
    n_trials: int,
) -> tuple[jax.Array, jax.Array]:
    """Trial types plus a balanced hidden-source assignment for probes."""
    key_order, key_hidden_source = jax.random.split(key)
    n_probes_per_source = spec.n_trials_partial // spec.n_sources

    trial_type = jnp.concatenate(
        [
            jnp.full((spec.n_trials_full,), TRUTH_ESTIMATION, dtype=jnp.int32),
            jnp.full((spec.n_trials_partial,), HIDDEN_SOURCE_PROBE, dtype=jnp.int32),
        ]
    )

    balanced_hidden_sources = jnp.tile(
        jnp.arange(spec.n_sources, dtype=jnp.int32),
        n_probes_per_source,
    )
    hidden_order = jax.random.permutation(key_hidden_source, spec.n_trials_partial)
    hidden_source = jnp.concatenate(
        [
            jnp.full((spec.n_trials_full,), NO_HIDDEN_SOURCE, dtype=jnp.int32),
            balanced_hidden_sources[hidden_order],
        ]
    )

    trial_order = jax.random.permutation(key_order, n_trials)
    return trial_type[trial_order], hidden_source[trial_order]


def _draw_until_resultant_floor(
    key: jax.Array,
    spec: StimulusSpec,
    n_trials: int,
    error_cholesky: jax.Array,
    resultant_floor: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Redraw only trials whose resultant is below the floor."""
    truth0 = jnp.zeros((n_trials,), dtype=error_cholesky.dtype)
    reports0 = jnp.zeros((n_trials, spec.n_sources), dtype=error_cholesky.dtype)
    resultant0 = jnp.full((n_trials,), -jnp.inf, dtype=error_cholesky.dtype)

    def draw_complete_reports(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        key_truth, key_error = jax.random.split(key)
        truth = jax.random.uniform(
            key_truth,
            (n_trials,),
            minval=0.0,
            maxval=2 * jnp.pi,
            dtype=error_cholesky.dtype,
        )
        error = (
            jax.random.normal(
                key_error,
                (n_trials, spec.n_sources),
                dtype=error_cholesky.dtype,
            )
            @ error_cholesky
        )
        reports = (truth[..., None] + error) % (2 * jnp.pi)
        return truth, reports

    def below_floor(resultant: jax.Array) -> jax.Array:
        return resultant < resultant_floor

    def keep_going(carry) -> jax.Array:
        _, attempts, _, _, resultant = carry
        return jnp.logical_and(
            jnp.any(below_floor(resultant)),
            attempts < _MAX_REJECTION_ATTEMPTS,
        )

    def attempt(carry):
        key, attempts, truth, reports, resultant = carry
        key, subkey = jax.random.split(key)
        new_truth, new_reports = draw_complete_reports(subkey)
        new_resultant = _resultant_length(new_reports)
        replace = below_floor(resultant)
        truth = jnp.where(replace, new_truth, truth)
        reports = jnp.where(replace[..., None], new_reports, reports)
        resultant = jnp.where(replace, new_resultant, resultant)
        return key, attempts + 1, truth, reports, resultant

    _, _, truth, reports, resultant = jax.lax.while_loop(
        keep_going,
        attempt,
        (key, jnp.int32(0), truth0, reports0, resultant0),
    )
    return truth, reports, resultant


def _resultant_length(reports: jax.Array) -> jax.Array:
    """Mean resultant length of the four arrows, in [0, 1].

    Near 1 when the arrows form one broad fan; near 0 when they split into two
    opposed clumps. Rotation-invariant, so it does not depend on the trial's truth.
    """
    x = jnp.cos(reports).sum(-1)
    y = jnp.sin(reports).sum(-1)
    return jnp.hypot(x, y) / reports.shape[-1]
