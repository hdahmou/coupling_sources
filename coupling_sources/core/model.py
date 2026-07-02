import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from coupling_sources.utils.angles import centered_angle
from coupling_sources.core.parameters.simulation import SimParams
from coupling_sources.core.response import compute_response_mean, compute_response_trace


def choice_model(
    simulator_output: dict[str, jax.Array | None],
    metadata: object | None = None,
    debug: bool = False,
) -> None:
    coupled_sources = (0, 1)
    if isinstance(metadata, dict) and "coupled_sources" in metadata:
        coupled_sources = tuple(metadata["coupled_sources"])

    source_choices = simulator_output["source_choices"]
    hidden_source = simulator_output["hidden_source"]
    choices = simulator_output["choices"]
    if debug:
        source_choices = numpyro.deterministic("source_reports_input", source_choices)
        hidden_source = numpyro.deterministic("hidden_source_input", hidden_source)

    priors = SimParams.prior_distributions()
    params = SimParams(*(numpyro.sample(name, priors[name]) for name in SimParams.names()))
    if debug:
        flow = compute_response_trace(params, source_choices, hidden_source)
        is_probe = numpyro.deterministic("is_probe_trial", flow.is_probe_trial)
        visible = numpyro.deterministic("visible_source_mask", flow.visible_source_mask)
        agreement_scores = numpyro.deterministic(
            "pre_update_agreement_scores", flow.pre_update_agreement_scores
        )
        numpyro.deterministic("source_weights", flow.source_weights)
        truth_mu = numpyro.deterministic(
            "truth_branch_mean_angle", flow.truth_branch_mean_angle
        )
        hidden_mu = numpyro.deterministic(
            "hidden_source_branch_mean_angle", flow.hidden_source_branch_mean_angle
        )
        response_mu = numpyro.deterministic(
            "selected_response_mean_angle", jnp.where(is_probe, hidden_mu, truth_mu)
        )
        numpyro.deterministic("hidden_source_index", flow.hidden_source_index)
        numpyro.deterministic("visible_source_count", visible.sum(axis=-1))
        numpyro.deterministic(
            "pre_update_coupled_agreement_score",
            agreement_scores[:, :, coupled_sources[0], coupled_sources[1]],
        )
        response_mu = numpyro.deterministic("response_mean_angle", flow.response_mean_angle)
    else:
        response_mu = compute_response_mean(params, source_choices, hidden_source)
    obs = None if choices is None else centered_angle(choices)

    with numpyro.plate("round", source_choices.shape[1], dim=-2):
        with numpyro.plate("trial", source_choices.shape[2], dim=-1):
            numpyro.sample(
                "response_angle",
                dist.VonMises(centered_angle(response_mu), params.kappa_p),
                obs=obs,
            )
