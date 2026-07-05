from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from coupling_sources.core.parameters.simulation import SimParams
from coupling_sources.utils.angles import centered_angle, circular_mean, optimal_weights


class ResponseTrace(NamedTuple):
    response_mean_angle: jax.Array
    truth_branch_mean_angle: jax.Array
    hidden_source_branch_mean_angle: jax.Array
    source_weights: jax.Array
    pre_update_agreement_scores: jax.Array
    post_update_agreement_scores: jax.Array
    visible_source_mask: jax.Array
    hidden_source_index: jax.Array
    is_probe_trial: jax.Array


def _visible_sources_for_trial(
    trial_reports: jax.Array, hidden_source: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_sources = trial_reports.shape[0]
    is_probe = ~jnp.isnan(hidden_source)
    hidden_index = jnp.asarray(jnp.nan_to_num(hidden_source, nan=0.0), dtype=jnp.int32)
    hidden_mask = jax.nn.one_hot(hidden_index, n_sources, dtype=jnp.bool_)
    visible_sources = (~jnp.isnan(trial_reports)) & ~(is_probe & hidden_mask)
    return visible_sources.astype(trial_reports.dtype), hidden_index, is_probe


def _prepare_trial(
    params: SimParams,
    n_sources: int,
    agreement_scores: jax.Array,
    trial_reports: jax.Array,
    trial_hidden_source: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    reports = jnp.nan_to_num(trial_reports, nan=0.0)
    visible_sources, hidden_index, is_probe = _visible_sources_for_trial(
        trial_reports, trial_hidden_source
    )
    source_covariance = jnp.eye(n_sources) + params.gamma * agreement_scores
    return reports, visible_sources, hidden_index, is_probe, source_covariance


def _predict_hidden_report(
    reports: jax.Array,
    source_covariance: jax.Array,
    visible_sources: jax.Array,
    hidden_index: jax.Array,
) -> jax.Array:
    visible_mean = circular_mean(reports, visible_sources)
    offsets = centered_angle(reports - visible_mean) * visible_sources
    visible_covariance = (
        source_covariance * jnp.outer(visible_sources, visible_sources)
        + jnp.diag(1.0 - visible_sources)
    )
    predicted_offset = source_covariance[hidden_index] @ jsp_linalg.solve(
        visible_covariance, offsets, assume_a="sym"
    )
    return (visible_mean + predicted_offset) % (2 * jnp.pi)


def _update_agreement_scores(
    agreement_scores: jax.Array,
    reports: jax.Array,
    visible_sources: jax.Array,
    params: SimParams,
) -> jax.Array:
    off_diagonal = 1.0 - jnp.eye(reports.shape[0])
    distance = reports[:, None] - reports[None, :]
    agreement = jnp.cos(distance)
    visible_pairs = jnp.outer(visible_sources, visible_sources) * off_diagonal
    return off_diagonal * (
        agreement_scores + params.alpha * (agreement - agreement_scores) * visible_pairs
    )


def _initial_agreement_scores(n_sources: int, c0: float | jax.Array) -> jax.Array:
    return (1.0 - jnp.eye(n_sources)) * c0


def _round_response_mean(
    params: SimParams, source_choices: jax.Array, hidden_source: jax.Array
) -> jax.Array:
    n_sources = source_choices.shape[0]

    def step(
        agreement_scores: jax.Array, trial: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        trial_reports, trial_hidden_source = trial
        reports, visible_sources, hidden_index, is_probe, source_covariance = _prepare_trial(
            params, n_sources, agreement_scores, trial_reports, trial_hidden_source
        )

        def hidden_response(_: None) -> jax.Array:
            return _predict_hidden_report(
                reports, source_covariance, visible_sources, hidden_index
            )

        def truth_response(_: None) -> jax.Array:
            source_weights = optimal_weights(source_covariance)
            return circular_mean(reports, source_weights * visible_sources)

        response_mean = jax.lax.cond(
            is_probe,
            hidden_response,
            truth_response,
            operand=None,
        )
        updated_scores = _update_agreement_scores(
            agreement_scores, reports, visible_sources, params
        )
        return updated_scores, response_mean

    _, response_mean = jax.lax.scan(
        step,
        _initial_agreement_scores(n_sources, params.c0),
        (source_choices.T, hidden_source),
    )
    return cast(jax.Array, response_mean)


def _round_response_trace(
    params: SimParams, source_choices: jax.Array, hidden_source: jax.Array
) -> ResponseTrace:
    n_sources = source_choices.shape[0]

    def step(
        agreement_scores: jax.Array, trial: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        trial_reports, trial_hidden_source = trial
        reports, visible_sources, hidden_index, is_probe, source_covariance = _prepare_trial(
            params, n_sources, agreement_scores, trial_reports, trial_hidden_source
        )
        source_weights = optimal_weights(source_covariance)
        truth_mean = circular_mean(reports, source_weights * visible_sources)

        def hidden_response(_: None) -> tuple[jax.Array, jax.Array]:
            hidden_mean = _predict_hidden_report(
                reports, source_covariance, visible_sources, hidden_index
            )
            return hidden_mean, hidden_mean

        def truth_response(_: None) -> tuple[jax.Array, jax.Array]:
            return jnp.full((), jnp.nan), truth_mean

        branch_response = cast(
            tuple[jax.Array, jax.Array],
            jax.lax.cond(
                is_probe,
                hidden_response,
                truth_response,
                operand=None,
            ),
        )
        hidden_mean, response_mean = branch_response
        updated_scores = _update_agreement_scores(
            agreement_scores, reports, visible_sources, params
        )
        return updated_scores, (
            response_mean,
            truth_mean,
            hidden_mean,
            source_weights,
            agreement_scores,
            updated_scores,
            visible_sources,
            hidden_index,
            is_probe,
        )

    _, out = jax.lax.scan(
        step,
        _initial_agreement_scores(n_sources, params.c0),
        (source_choices.T, hidden_source),
    )
    (
        response_mean,
        truth_mean,
        hidden_mean,
        source_weights,
        pre_update_scores,
        post_update_scores,
        visible_sources,
        hidden_index,
        is_probe,
    ) = cast(tuple[jax.Array, ...], out)
    return ResponseTrace(
        response_mean_angle=response_mean,
        truth_branch_mean_angle=truth_mean,
        hidden_source_branch_mean_angle=hidden_mean,
        source_weights=source_weights,
        pre_update_agreement_scores=pre_update_scores,
        post_update_agreement_scores=post_update_scores,
        visible_source_mask=visible_sources,
        hidden_source_index=hidden_index,
        is_probe_trial=is_probe,
    )


def compute_response_trace(
    params: SimParams, source_choices: jax.Array, hidden_source: jax.Array
) -> ResponseTrace:
    return cast(
        ResponseTrace,
        jax.vmap(_round_response_trace, in_axes=(None, 1, 0))(
            params, source_choices, hidden_source
        ),
    )


def compute_response_mean(
    params: SimParams, source_choices: jax.Array, hidden_source: jax.Array
) -> jax.Array:
    return cast(
        jax.Array,
        jax.vmap(_round_response_mean, in_axes=(None, 1, 0))(
            params, source_choices, hidden_source
        ),
    )
