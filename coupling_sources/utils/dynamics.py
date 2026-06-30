import jax
import jax.numpy as jnp

from coupling_sources.utils.angles import centered_angle, circular_mean, optimal_weights
from coupling_sources.parameters.simulation import SimParams
from coupling_sources.utils.shapes import as_source_round_data, source_round_to_round_source


def trial_visibility(
    reports_t: jax.Array, hidden_t: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_sources = reports_t.shape[0]
    is_probe = ~jnp.isnan(hidden_t)
    hidden_idx = jnp.asarray(jnp.nan_to_num(hidden_t, nan=0.0), dtype=jnp.int32)
    visible = (~jnp.isnan(reports_t)) & (
        ~(is_probe & (jnp.arange(n_sources) == hidden_idx))
    )
    return visible.astype(float), hidden_idx, is_probe


def predict_hidden_source(
    reports: jax.Array, working_cov: jax.Array, visible: jax.Array, hidden_idx: jax.Array
) -> jax.Array:
    visible_mean = circular_mean(reports, visible)
    offsets = centered_angle(reports - visible_mean) * visible
    visible_cov = (
        working_cov * visible[:, None] * visible[None, :]
        + jnp.diag(1.0 - visible)
    )
    return (visible_mean + working_cov[hidden_idx] @ jnp.linalg.solve(visible_cov, offsets)) % (2 * jnp.pi)


def update_agreement_scores(
    agreement_scores: jax.Array, reports: jax.Array, visible: jax.Array, params: SimParams
) -> jax.Array:
    off_diag = 1.0 - jnp.eye(reports.shape[0])
    distance = reports[:, None] - reports[None, :]
    agreement = jnp.exp(params.kappa_a * (jnp.cos(distance) - 1.0))
    visible_pair = visible[:, None] * visible[None, :] * off_diag
    return off_diag * (agreement_scores + params.alpha * (agreement - agreement_scores) * visible_pair)


def round_flow(
    params: SimParams, source_choices: jax.Array, hidden_source: jax.Array
) -> dict[str, jax.Array]:
    n_sources = source_choices.shape[0]

    def step(
        agreement_scores: jax.Array, trial: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        reports_t, hidden_t = trial
        reports = jnp.nan_to_num(reports_t, nan=0.0)
        visible, hidden_idx, is_probe = trial_visibility(reports_t, hidden_t)
        working_cov = jnp.eye(n_sources) + params.gamma * agreement_scores
        weights = optimal_weights(working_cov)

        truth_mu = circular_mean(reports, weights * visible)
        hidden_mu = predict_hidden_source(reports, working_cov, visible, hidden_idx)
        response_mu = jnp.where(is_probe, hidden_mu, truth_mu)
        updated_scores = update_agreement_scores(agreement_scores, reports, visible, params)
        return updated_scores, (
            response_mu,
            truth_mu,
            hidden_mu,
            weights,
            agreement_scores,
            updated_scores,
            visible,
            hidden_idx,
            is_probe,
        )

    _, out = jax.lax.scan(
        step, jnp.zeros((n_sources, n_sources)), (source_choices.T, hidden_source)
    )
    (
        response_mu,
        truth_mu,
        hidden_mu,
        weights,
        pre_update_scores,
        post_update_scores,
        visible,
        hidden_idx,
        is_probe,
    ) = out
    return {
        "response_mean_angle": response_mu,
        "truth_branch_mean_angle": truth_mu,
        "hidden_source_branch_mean_angle": hidden_mu,
        "source_weights": weights,
        "pre_update_agreement_scores": pre_update_scores,
        "post_update_agreement_scores": post_update_scores,
        "visible_source_mask": visible,
        "hidden_source_index": hidden_idx,
        "is_probe_trial": is_probe,
    }


def response_flow(
    params: SimParams, source_choices: jax.Array, hidden_source: jax.Array
) -> dict[str, jax.Array]:
    source_choices, hidden_source, _ = as_source_round_data(source_choices, hidden_source)
    return jax.vmap(lambda choices_i, hidden_i: round_flow(params, choices_i, hidden_i))(
        source_round_to_round_source(source_choices), hidden_source
    )
