from __future__ import annotations

import jax
import numpyro

from coupling_sources.model import DEFAULT_PRIOR_SIMULATOR_OUTPUT, PARAMETER_SITES, choice_model
from coupling_sources.parameters.simulation import SimParams
from coupling_sources.utils.posterior import extract_parameter_samples


def sample_parameter_prior(seed: int, draws: int) -> dict[str, jax.Array]:
    if draws < 1:
        raise ValueError("prior draws must be at least 1")

    predictive = numpyro.infer.Predictive(
        choice_model,
        num_samples=draws,
        return_sites=tuple(PARAMETER_SITES.values()),
    )
    prior_draw = predictive(jax.random.PRNGKey(seed), DEFAULT_PRIOR_SIMULATOR_OUTPUT)
    return extract_parameter_samples(prior_draw)


def sample_prior_parameter_sets(seed: int, draws: int) -> list[SimParams]:
    samples = sample_parameter_prior(seed, draws)
    return [
        SimParams.from_mapping({parameter: values[index] for parameter, values in samples.items()})
        for index in range(draws)
    ]
