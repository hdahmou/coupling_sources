from __future__ import annotations

import jax
import jax.numpy as jnp

from coupling_sources.core.constants import (
    HIDDEN_SOURCE_PROBE,
    NO_HIDDEN_SOURCE,
    TRUTH_ESTIMATION,
)
from coupling_sources.core.data import StimulusBatch, StimulusMetadata, StimulusSpec

_MAX_REJECTION_ATTEMPTS = 10_000


def generate_stimulus(spec: StimulusSpec) -> StimulusBatch:
    n_trials = spec.n_trials_full + spec.n_trials_partial

    corr = jnp.eye(spec.n_sources)
    a, b = spec.coupled_sources
    corr = corr.at[a, b].set(spec.lam).at[b, a].set(spec.lam)
    covariance = spec.sigma**2 * corr
    error_cholesky = jnp.linalg.cholesky(covariance).T
    resultant_floor_full = jnp.asarray(
        spec.resultant_floor_full,
        dtype=covariance.dtype,
    )
    resultant_floor_leave_one_out = jnp.asarray(
        spec.resultant_floor_leave_one_out,
        dtype=covariance.dtype,
    )

    base_key = jax.random.PRNGKey(spec.seed)
    rounds = jnp.arange(spec.n_rounds)

    def keys_for_participant(participant: jax.Array) -> jax.Array:
        participant_key = jax.random.fold_in(base_key, participant)
        return jax.vmap(lambda round_: jax.random.fold_in(participant_key, round_))(
            rounds
        )

    round_keys = jax.vmap(keys_for_participant)(jnp.arange(spec.n_participants))

    def generate_round(key: jax.Array) -> tuple[jax.Array, ...]:
        key_trials, key_schedule = jax.random.split(key)
        (
            truth,
            complete_reports,
            resultant_length_full,
            resultant_length_leave_one_out,
        ) = _draw_until_resultant_floors(
            key_trials,
            spec,
            n_trials,
            error_cholesky,
            resultant_floor_full,
            resultant_floor_leave_one_out,
        )
        trial_type, hidden_source = _draw_trial_schedule(key_schedule, spec, n_trials)

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
            resultant_length_full,
            resultant_length_leave_one_out,
        )

    flat_rounds = jax.lax.map(
        generate_round,
        round_keys.reshape((-1, 2)),
    )
    round_grids = tuple(
        value.reshape((spec.n_participants, spec.n_rounds, *value.shape[1:]))
        for value in flat_rounds
    )
    resultant_length_full = round_grids[-2]
    resultant_length_leave_one_out = round_grids[-1]

    if n_trials > 0:
        lowest_full = float(jax.device_get(jnp.min(resultant_length_full)))
        lowest_leave_one_out = float(
            jax.device_get(jnp.min(resultant_length_leave_one_out))
        )
        if (
            lowest_full < spec.resultant_floor_full
            or lowest_leave_one_out < spec.resultant_floor_leave_one_out
        ):
            raise RuntimeError(
                f"could not satisfy resultant floors after {_MAX_REJECTION_ATTEMPTS} "
                f"attempts; lowest full resultant was {lowest_full:.6g} "
                f"(floor {spec.resultant_floor_full:.6g}) and lowest leave-one-out "
                f"resultant was {lowest_leave_one_out:.6g} "
                f"(floor {spec.resultant_floor_leave_one_out:.6g})"
            )

    (
        visible_reports,
        hidden_source,
        trial_type,
        complete_reports,
        truth,
        resultant_length_full,
        resultant_length_leave_one_out,
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
            resultant_length_full=resultant_length_full,
            resultant_length_leave_one_out=resultant_length_leave_one_out,
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


def _draw_until_resultant_floors(
    key: jax.Array,
    spec: StimulusSpec,
    n_trials: int,
    error_cholesky: jax.Array,
    resultant_floor_full: jax.Array,
    resultant_floor_leave_one_out: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Redraw complete displays that fail either resultant floor."""
    truth0 = jnp.zeros((n_trials,), dtype=error_cholesky.dtype)
    reports0 = jnp.zeros((n_trials, spec.n_sources), dtype=error_cholesky.dtype)
    resultant_full0 = jnp.full(
        (n_trials,),
        -jnp.inf,
        dtype=error_cholesky.dtype,
    )
    resultant_leave_one_out0 = jnp.full(
        (n_trials, spec.n_sources),
        -jnp.inf,
        dtype=error_cholesky.dtype,
    )

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

    def below_floor(
        resultant_full: jax.Array,
        resultant_leave_one_out: jax.Array,
    ) -> jax.Array:
        return (resultant_full < resultant_floor_full) | jnp.any(
            resultant_leave_one_out < resultant_floor_leave_one_out,
            axis=-1,
        )

    def keep_going(carry) -> jax.Array:
        _, attempts, _, _, resultant_full, resultant_leave_one_out = carry
        return jnp.logical_and(
            jnp.any(below_floor(resultant_full, resultant_leave_one_out)),
            attempts < _MAX_REJECTION_ATTEMPTS,
        )

    def attempt(carry):
        (
            key,
            attempts,
            truth,
            reports,
            resultant_full,
            resultant_leave_one_out,
        ) = carry
        key, subkey = jax.random.split(key)
        new_truth, new_reports = draw_complete_reports(subkey)
        new_resultant_full, new_resultant_leave_one_out = _resultant_lengths(
            new_reports
        )
        replace = below_floor(resultant_full, resultant_leave_one_out)
        truth = jnp.where(replace, new_truth, truth)
        reports = jnp.where(replace[..., None], new_reports, reports)
        resultant_full = jnp.where(replace, new_resultant_full, resultant_full)
        resultant_leave_one_out = jnp.where(
            replace[..., None],
            new_resultant_leave_one_out,
            resultant_leave_one_out,
        )
        return (
            key,
            attempts + 1,
            truth,
            reports,
            resultant_full,
            resultant_leave_one_out,
        )

    (
        _,
        _,
        truth,
        reports,
        resultant_full,
        resultant_leave_one_out,
    ) = jax.lax.while_loop(
        keep_going,
        attempt,
        (
            key,
            jnp.int32(0),
            truth0,
            reports0,
            resultant_full0,
            resultant_leave_one_out0,
        ),
    )
    return truth, reports, resultant_full, resultant_leave_one_out


def _resultant_lengths(reports: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Full and leave-one-out mean resultant lengths, each in [0, 1].

    Near 1 when the arrows form one broad fan; near 0 when they split into two
    opposed clumps. Rotation-invariant, so it does not depend on the trial's truth.
    """
    cos_reports = jnp.cos(reports)
    sin_reports = jnp.sin(reports)
    x = cos_reports.sum(-1)
    y = sin_reports.sum(-1)
    n_sources = reports.shape[-1]
    full = jnp.hypot(x, y) / n_sources
    leave_one_out = jnp.hypot(
        x[..., None] - cos_reports,
        y[..., None] - sin_reports,
    ) / (n_sources - 1)
    return full, leave_one_out
