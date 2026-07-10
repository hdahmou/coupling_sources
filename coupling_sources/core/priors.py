"""Prior distributions and parameter helpers for the coupling model."""

from __future__ import annotations

from collections.abc import Mapping
from math import log
from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import transforms


class CouplingPriors:
    """Static namespace for coupling-model prior and parameter helpers."""

    PARAMETER_SITES = ("learningRate", "g_y", "g_z", "vmConcentration")
    SIMULATION_PARAMETER_SITES = (*PARAMETER_SITES, "c0")

    @staticmethod
    def distributions() -> dict[str, dist.Distribution]:
        return {
            "learningRate": dist.TransformedDistribution(
                dist.Normal(0.0, 1.5),
                transforms.SigmoidTransform(),
            ),
            "g_y": dist.LogNormal(log(8.0), 1.0),
            "g_z": dist.LogNormal(log(8.0), 1.0),
            "vmConcentration": dist.LogNormal(log(8.0), 1.0),
        }

    @staticmethod
    def parameter_sites() -> tuple[str, ...]:
        return CouplingPriors.PARAMETER_SITES

    @staticmethod
    def simulation_parameter_sites() -> tuple[str, ...]:
        return CouplingPriors.SIMULATION_PARAMETER_SITES

    @staticmethod
    def normalize_parameters(values: Mapping[str, Any]) -> dict[str, jax.Array]:
        params = {
            name: jnp.asarray(values[name], dtype=jnp.float64)
            for name in CouplingPriors.PARAMETER_SITES
        }
        params["c0"] = jnp.asarray(values["c0"], dtype=jnp.float64)
        return params

    @staticmethod
    def parameters_as_floats(values: Mapping[str, Any]) -> dict[str, float]:
        params = CouplingPriors.normalize_parameters(values)
        return {
            name: float(jnp.asarray(params[name]))
            for name in CouplingPriors.SIMULATION_PARAMETER_SITES
        }

    @staticmethod
    def sample_parameter_draws(
        seed: int,
        draws: int,
        c0: float | jax.Array,
    ) -> dict[str, jax.Array]:
        priors = CouplingPriors.distributions()
        keys = jax.random.split(jax.random.PRNGKey(seed), len(priors))
        samples = {
            name: jnp.asarray(prior.sample(key, sample_shape=(draws,)))
            for key, (name, prior) in zip(keys, priors.items(), strict=True)
        }
        samples["c0"] = jnp.broadcast_to(
            jnp.asarray(c0, dtype=jnp.float64),
            (draws,),
        )
        return samples
