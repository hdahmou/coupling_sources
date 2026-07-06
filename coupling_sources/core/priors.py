"""Prior distributions and parameter helpers for the coupling model."""

from __future__ import annotations

from collections.abc import Mapping
from math import log
from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import transforms

COUPLING_PARAMETER_SITES = ("alpha", "beta_y", "beta_z", "kappa")
COUPLING_SIMULATION_PARAMETER_SITES = (*COUPLING_PARAMETER_SITES, "c0")


def coupling_prior_distributions() -> dict[str, dist.Distribution]:
    return {
        "alpha": dist.TransformedDistribution(
            dist.Normal(0.0, 1.5),
            transforms.SigmoidTransform(),
        ),
        "beta_y": dist.TransformedDistribution(
            dist.Normal(0.0, 1.0),
            [
                transforms.SigmoidTransform(),
                transforms.AffineTransform(loc=0.0, scale=10.0),
            ],
        ),
        "beta_z": dist.TransformedDistribution(
            dist.Normal(0.0, 1.0),
            [
                transforms.SigmoidTransform(),
                transforms.AffineTransform(loc=0.0, scale=10.0),
            ],
        ),
        "kappa": dist.LogNormal(log(8.0), 1.0),
    }


def coupling_parameter_sites() -> tuple[str, ...]:
    return COUPLING_PARAMETER_SITES


def coupling_simulation_parameter_sites() -> tuple[str, ...]:
    return COUPLING_SIMULATION_PARAMETER_SITES


def normalize_coupling_parameters(values: Mapping[str, Any]) -> dict[str, jax.Array]:
    params = {name: jnp.asarray(values[name]) for name in COUPLING_PARAMETER_SITES}
    params["c0"] = jnp.asarray(values.get("c0", 0.0))
    return params


def coupling_parameters_as_floats(values: Mapping[str, Any]) -> dict[str, float]:
    params = normalize_coupling_parameters(values)
    return {name: float(jnp.asarray(params[name])) for name in COUPLING_SIMULATION_PARAMETER_SITES}


def sample_coupling_parameter_draws(seed: int, draws: int) -> dict[str, jax.Array]:
    priors = coupling_prior_distributions()
    keys = jax.random.split(jax.random.PRNGKey(seed), len(priors))
    samples = {
        name: jnp.asarray(prior.sample(key, sample_shape=(draws,)))
        for key, (name, prior) in zip(keys, priors.items(), strict=True)
    }
    samples["c0"] = jnp.zeros((draws,))
    return samples
