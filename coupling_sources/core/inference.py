from __future__ import annotations

import os
import re
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.diagnostics as diagnostics
import numpyro
import numpyro.infer.mcmc as numpyro_mcmc
import numpyro.util as numpyro_util
import pandas as pd

from coupling_sources.core.model import choice_model
from coupling_sources.core.parameters.mcmc import MCMCParams
from coupling_sources.core.parameters.simulation import SimParams
from coupling_sources.core.simulation import SimulationArtifact
from coupling_sources.core.summarize_posterior import (
    parameter_values,
    summarize_posterior,
)


def fit_mcmc(
    key: jax.Array,
    choices: jax.Array,
    source_choices: jax.Array,
    hidden_source: jax.Array,
    num_warmup: int = 300,
    num_samples: int = 600,
    num_chains: int = 1,
    chain_method: str = "parallel",
    target_accept_prob: float = 0.85,
    max_tree_depth: int = 10,
    progress_bar: bool = False,
) -> numpyro.infer.MCMC:
    data = {
        "choices": choices,
        "source_choices": source_choices,
        "hidden_source": hidden_source,
    }
    kernel = numpyro.infer.NUTS(
        choice_model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    mcmc.run(key, data, None, extra_fields=("diverging",))
    return mcmc


class _LabeledTqdm:
    def __init__(self, bar: object, label: str):
        self.bar = bar
        self.label = label

    def set_description(self, description: object, *args: object, **kwargs: object) -> object:
        text = str(description).strip()
        labeled = f"[{self.label}] {text}" if text else f"[{self.label}]"
        return self.bar.set_description(labeled, *args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self.bar, name)


@contextmanager
def _labeled_progress_bar(label: str | None):
    if label is None:
        yield
        return

    original_fori_collect = numpyro_mcmc.fori_collect
    original_tqdm_auto = numpyro_util.tqdm_auto

    def with_label(description: object) -> str:
        text = str(description).strip()
        return f"[{label}] {text}" if text else f"[{label}]"

    def fori_collect_with_label(*args: object, **kwargs: object) -> object:
        original_desc = kwargs.get("progbar_desc", lambda _: "")

        def progbar_desc(index: object) -> str:
            return with_label(original_desc(index))

        kwargs["progbar_desc"] = progbar_desc
        return original_fori_collect(*args, **kwargs)

    def tqdm_auto_with_label(*args: object, **kwargs: object) -> _LabeledTqdm:
        return _LabeledTqdm(original_tqdm_auto(*args, **kwargs), label)

    numpyro_mcmc.fori_collect = fori_collect_with_label
    numpyro_util.tqdm_auto = tqdm_auto_with_label
    try:
        yield
    finally:
        numpyro_mcmc.fori_collect = original_fori_collect
        numpyro_util.tqdm_auto = original_tqdm_auto


def _configure_parallel_chains(
    *, num_chains: int, chain_method: str, host_device_count: int | None = None
) -> None:
    effective = host_device_count
    if effective is None and chain_method == "parallel" and num_chains > 1:
        effective = num_chains
    if effective is None:
        return
    if effective < 1:
        raise ValueError("host_device_count must be a positive integer")

    flags = os.environ.get("XLA_FLAGS", "")
    flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", flags).split()
    os.environ["XLA_FLAGS"] = " ".join(
        [f"--xla_force_host_platform_device_count={effective}", *flags]
    )


def _mcmc_diagnostics(mcmc: object) -> dict[str, object]:
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    parameter_names = SimParams.names()
    parameter_summary = diagnostics.summary(
        {name: grouped_samples[name] for name in parameter_names},
        group_by_chain=True,
    )
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)
    divergences = extra_fields.get("diverging")
    n_divergences = None if divergences is None else int(np.asarray(divergences).sum())
    return {
        "parameters": {
            parameter: {
                "r_hat": float(np.asarray(stats["r_hat"])),
                "ess": float(np.asarray(stats["n_eff"])),
            }
            for parameter, stats in parameter_summary.items()
        },
        "divergences": {
            "has_divergences": None if n_divergences is None else bool(n_divergences),
            "n_divergences": n_divergences,
        },
    }


def fit_posterior(
    *,
    simulation: SimulationArtifact,
    params: MCMCParams,
    host_device_count: int | None,
    progress_bar: bool,
    progress_label: str | None = None,
) -> dict[str, object]:
    _configure_parallel_chains(
        num_chains=params.num_chains,
        chain_method=params.chain_method,
        host_device_count=host_device_count,
    )

    choices = jnp.asarray(simulation.choices)
    source_choices = jnp.asarray(simulation.source_choices)
    hidden_source = jnp.asarray(simulation.hidden_source)
    true_params = simulation.participant_parameters

    with _labeled_progress_bar(progress_label if progress_bar else None):
        mcmc = fit_mcmc(
            jax.random.PRNGKey(params.seed),
            choices,
            source_choices,
            hidden_source,
            num_warmup=params.num_warmup,
            num_samples=params.num_samples,
            num_chains=params.num_chains,
            chain_method=params.chain_method,
            target_accept_prob=params.target_accept_prob,
            max_tree_depth=params.max_tree_depth,
            progress_bar=progress_bar,
        )
    flat_samples = mcmc.get_samples(group_by_chain=False)
    chain_samples = mcmc.get_samples(group_by_chain=True)
    parameter_names = SimParams.names()
    summary = summarize_posterior(
        pd.DataFrame({name: np.asarray(flat_samples[name]) for name in parameter_names}),
        parameter_values(true_params),
    )
    summary.index.name = "parameter"
    return {
        "chain_method": mcmc.chain_method,
        "diagnostics": _mcmc_diagnostics(mcmc),
        "samples": {name: chain_samples[name] for name in parameter_names},
        "summary": summary,
    }
