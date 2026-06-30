import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from coupling_sources.utils.angles import centered_angle
from coupling_sources.utils.dynamics import response_flow
from coupling_sources.parameters.simulation import SimParams
from coupling_sources.utils.shapes import as_source_round_data


PARAMETER_SITES = {
    "alpha": "participant_alpha",
    "gamma": "participant_gamma",
    "kappa_p": "participant_response_concentration",
    "kappa_a": "agreement_kernel_concentration",
}

DEFAULT_PRIOR_SIMULATOR_OUTPUT = {
    "choices": None,
    "source_choices": np.zeros((2, 1, 1)),
    "hidden_source": np.full((1, 1), np.nan),
}


def choice_model(
    simulator_output: dict[str, jax.Array | None],
    metadata: object | None = None,
    debug: bool = False,
) -> None:
    coupled_sources = (0, 1)
    if isinstance(metadata, dict) and "coupled_sources" in metadata:
        coupled_sources = tuple(metadata["coupled_sources"])

    source_choices, hidden_source, choices = as_source_round_data(
        simulator_output["source_choices"],
        simulator_output["hidden_source"],
        simulator_output["choices"],
    )
    if debug:
        source_choices = numpyro.deterministic("source_reports_input", source_choices)
        hidden_source = numpyro.deterministic("hidden_source_input", hidden_source)

    params = SimParams(
        alpha=numpyro.sample(PARAMETER_SITES["alpha"], dist.Beta(2.0, 2.0)),
        gamma=numpyro.sample(PARAMETER_SITES["gamma"], dist.Beta(2.0, 2.0)),
        kappa_p=numpyro.sample(
            PARAMETER_SITES["kappa_p"], dist.LogNormal(jnp.log(8.0), 1.0)
        ),
        kappa_a=numpyro.sample(
            PARAMETER_SITES["kappa_a"], dist.LogNormal(jnp.log(5.0), 0.5)
        ),
    )
    flow = response_flow(params, source_choices, hidden_source)
    response_mu = flow["response_mean_angle"]

    if debug:
        is_probe = numpyro.deterministic("is_probe_trial", flow["is_probe_trial"])
        visible = numpyro.deterministic("visible_source_mask", flow["visible_source_mask"])
        agreement_scores = numpyro.deterministic(
            "pre_update_agreement_scores", flow["pre_update_agreement_scores"]
        )
        numpyro.deterministic("source_weights", flow["source_weights"])
        truth_mu = numpyro.deterministic(
            "truth_branch_mean_angle", flow["truth_branch_mean_angle"]
        )
        hidden_mu = numpyro.deterministic(
            "hidden_source_branch_mean_angle", flow["hidden_source_branch_mean_angle"]
        )
        response_mu = numpyro.deterministic(
            "selected_response_mean_angle", jnp.where(is_probe, hidden_mu, truth_mu)
        )
        numpyro.deterministic("hidden_source_index", flow["hidden_source_index"])
        numpyro.deterministic("visible_source_count", visible.sum(axis=-1))
        numpyro.deterministic(
            "pre_update_coupled_agreement_score",
            agreement_scores[:, :, coupled_sources[0], coupled_sources[1]],
        )

    response_mu = numpyro.deterministic("response_mean_angle", response_mu)
    obs = None if choices is None else centered_angle(choices)

    with numpyro.plate("round", source_choices.shape[1], dim=-2):
        with numpyro.plate("trial", source_choices.shape[2], dim=-1):
            numpyro.sample(
                "response_angle",
                dist.VonMises(centered_angle(response_mu), params.kappa_p),
                obs=obs,
            )


def fixed_participant_samples(params: SimParams) -> dict[str, jax.Array]:
    return {
        PARAMETER_SITES["alpha"]: jnp.reshape(params.alpha, (1,)),
        PARAMETER_SITES["gamma"]: jnp.reshape(params.gamma, (1,)),
        PARAMETER_SITES["kappa_p"]: jnp.reshape(params.kappa_p, (1,)),
        PARAMETER_SITES["kappa_a"]: jnp.reshape(params.kappa_a, (1,)),
    }
