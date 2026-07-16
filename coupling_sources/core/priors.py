"""Prior distributions and parameter helpers for the coupling model."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, log
from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import transforms


class CouplingPriors:
    """Static namespace for coupling-model prior and parameter helpers."""

    PARAMETER_SITES = ("learningRate", "gamma_y", "gamma_z", "vmConcentration")
    SIMULATION_PARAMETER_SITES = (*PARAMETER_SITES, "c0")

    @staticmethod
    def distributions() -> dict[str, dist.Distribution]:
        return {
            "learningRate": dist.TransformedDistribution(
                dist.Normal(-2.0, 2.0),
                transforms.SigmoidTransform(),
            ),
            "gamma_y": dist.Normal(0.0, 1.0),
            "gamma_z": dist.Normal(0.0, 1.0),
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
        params = {}
        for name in CouplingPriors.SIMULATION_PARAMETER_SITES:
            value = jnp.asarray(values[name], dtype=jnp.float64)
            if value.ndim != 0:
                raise ValueError(f"{name} must be a scalar")
            scalar = float(value)
            if not isfinite(scalar):
                raise ValueError(f"{name} must be finite")
            params[name] = value

        learning_rate = float(params["learningRate"])
        if not 0.0 < learning_rate < 1.0:
            raise ValueError("learningRate must be strictly between 0 and 1")
        if float(params["vmConcentration"]) <= 0.0:
            raise ValueError("vmConcentration must be positive")
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
