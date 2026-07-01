from __future__ import annotations

import jax
import jax.numpy as jnp

from coupling_sources.core.parameters.simulation import SimParams


def sample_parameter_prior(seed: int, draws: int) -> dict[str, jax.Array]:
    if draws < 1:
        raise ValueError("prior draws must be at least 1")

    priors = SimParams.prior_distributions()
    parameter_names = SimParams.names()
    keys = jax.random.split(jax.random.PRNGKey(seed), len(parameter_names))
    return {
        name: jnp.asarray(priors[name].sample(key, sample_shape=(draws,)))
        for key, name in zip(keys, parameter_names, strict=True)
    }
