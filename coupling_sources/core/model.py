from typing import Annotated

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from coupling_sources.utils.angles import centered_angle
from coupling_sources.core.parameters.simulation import SimParams
from coupling_sources.core.response import compute_response_mean, compute_response_trace


def choice_model(
    choices: (
        Annotated[
            jax.Array,
            "shape: (round, trial), observed response angle in radians",
        ]
        | None
    ),
    source_choices: Annotated[
        jax.Array,
        "shape: (source, round, trial), source report angle in radians; NaN when hidden",
    ],
    hidden_source: Annotated[
        jax.Array,
        "shape: (round, trial), hidden source index on probe trials; NaN otherwise",
    ],
    debug: bool = False,
) -> None:
    params = SimParams.sample_prior()
    if debug:
        numpyro.deterministic("source_reports_input", source_choices)
        numpyro.deterministic("hidden_source_input", hidden_source)
        flow = compute_response_trace(params, source_choices, hidden_source)
        is_probe = flow.is_probe_trial
        visible = flow.visible_source_mask
        truth_mu = flow.truth_branch_mean_angle
        hidden_mu = flow.hidden_source_branch_mean_angle
        response_mu = flow.response_mean_angle

        numpyro.deterministic("is_probe_trial", is_probe)
        numpyro.deterministic("visible_source_mask", visible)
        numpyro.deterministic(
            "pre_update_agreement_scores", flow.pre_update_agreement_scores
        )
        numpyro.deterministic("source_weights", flow.source_weights)
        numpyro.deterministic("truth_branch_mean_angle", truth_mu)
        numpyro.deterministic("hidden_source_branch_mean_angle", hidden_mu)
        numpyro.deterministic(
            "selected_response_mean_angle", jnp.where(is_probe, hidden_mu, truth_mu)
        )
        numpyro.deterministic("hidden_source_index", flow.hidden_source_index)
        numpyro.deterministic("visible_source_count", visible.sum(axis=-1))
        numpyro.deterministic("response_mean_angle", response_mu)
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
