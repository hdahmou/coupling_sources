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

    PARAMETER_SITES = ("α", "g_y", "g_z", "κ")
    SIMULATION_PARAMETER_SITES = (*PARAMETER_SITES, "c0")

    @staticmethod
    def distributions() -> dict[str, dist.Distribution]:
        return {
            "α": dist.TransformedDistribution(
                dist.Normal(0.0, 1.5),
                transforms.SigmoidTransform(),
            ),
            "g_y": dist.LogNormal(log(8.0), 1.0),
            "g_z": dist.LogNormal(log(8.0), 1.0),
            "κ": dist.LogNormal(log(8.0), 1.0),
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
        params["c0"] = jnp.asarray(values.get("c0", 0.0), dtype=jnp.float64)
        return params

    @staticmethod
    def parameters_as_floats(values: Mapping[str, Any]) -> dict[str, float]:
        params = CouplingPriors.normalize_parameters(values)
        return {
            name: float(jnp.asarray(params[name]))
            for name in CouplingPriors.SIMULATION_PARAMETER_SITES
        }

    @staticmethod
    def sample_parameter_draws(seed: int, draws: int) -> dict[str, jax.Array]:
        priors = CouplingPriors.distributions()
        keys = jax.random.split(jax.random.PRNGKey(seed), len(priors))
        samples = {
            name: jnp.asarray(prior.sample(key, sample_shape=(draws,)))
            for key, (name, prior) in zip(keys, priors.items(), strict=True)
        }
        samples["c0"] = jnp.zeros((draws,), dtype=jnp.float64)
        return samples
