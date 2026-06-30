from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.stats import vonmises

from coupling_sources.utils.angles import centered_angle
from coupling_sources.utils.dynamics import response_flow
from coupling_sources.parameters.simulation import SimParams
from coupling_sources.utils.shapes import as_source_round_data
from coupling_sources.utils.vonmises import sample_vonmises


@dataclass(frozen=True)
class SimulationTrace:
    response_mean: jax.Array
    truth_mean: jax.Array
    hidden_mean: jax.Array
    source_weights: jax.Array
    pre_update_agreement_scores: jax.Array
    choice_logpdf: jax.Array
    is_probe: jax.Array
    hidden_source: jax.Array


@dataclass(frozen=True)
class SimulationResult:
    choices: jax.Array
    trace: SimulationTrace


def simulate_choices(
    key: jax.Array,
    params: SimParams,
    source_choices: jax.Array,
    hidden_source: jax.Array,
) -> SimulationResult:
    source_choices, hidden_source, _ = as_source_round_data(source_choices, hidden_source)
    flow = response_flow(params, source_choices, hidden_source)
    n_rounds, n_trials = hidden_source.shape
    keys = jax.random.split(key, n_rounds * n_trials).reshape(n_rounds, n_trials, 2)
    choices = jax.vmap(
        lambda keys_i, mu_i: jax.vmap(
            lambda subkey, mu_t: sample_vonmises(subkey, mu_t, params.kappa_p)
        )(keys_i, mu_i)
    )(keys, flow["response_mean_angle"])

    trace = SimulationTrace(
        response_mean=flow["response_mean_angle"],
        truth_mean=flow["truth_branch_mean_angle"],
        hidden_mean=flow["hidden_source_branch_mean_angle"],
        source_weights=flow["source_weights"],
        pre_update_agreement_scores=flow["pre_update_agreement_scores"],
        choice_logpdf=vonmises.logpdf(
            centered_angle(choices - flow["response_mean_angle"]), params.kappa_p
        ),
        is_probe=flow["is_probe_trial"],
        hidden_source=hidden_source,
    )
    return SimulationResult(choices=choices, trace=trace)
