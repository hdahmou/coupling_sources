from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.diagnostics as diagnostics
import numpyro.infer.mcmc as numpyro_mcmc
import numpyro.util as numpyro_util

from coupling_sources.inference import fit_mcmc
from coupling_sources.parameters.mcmc import MCMCParams
from coupling_sources.parameters.simulation import SimParams
from artifacts import read_simulation_zarr, write_posterior_zarr
from coupling_sources.utils.posterior import (
    extract_parameter_samples,
    parameter_values,
    posterior_frame,
    summarize_posterior,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit one posterior artifact from one simulation.")
    parser.add_argument("--simulation", type=Path, required=True, help="Simulation Zarr directory.")
    MCMCParams.add_arguments(parser)
    parser.add_argument("--host-device-count", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-label", default=None)
    parser.add_argument("--output", type=Path, required=True, help="Output Zarr directory.")
    return parser.parse_args(argv)


class LabeledTqdm:
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
def labeled_progress_bar(label: str | None):
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

    def tqdm_auto_with_label(*args: object, **kwargs: object) -> LabeledTqdm:
        return LabeledTqdm(original_tqdm_auto(*args, **kwargs), label)

    numpyro_mcmc.fori_collect = fori_collect_with_label
    numpyro_util.tqdm_auto = tqdm_auto_with_label
    try:
        yield
    finally:
        numpyro_mcmc.fori_collect = original_fori_collect
        numpyro_util.tqdm_auto = original_tqdm_auto


def configure_parallel_chains(
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


def mcmc_diagnostics(mcmc: object) -> dict[str, object]:
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    parameter_summary = diagnostics.summary(
        extract_parameter_samples(grouped_samples),
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
    simulation: dict[str, object],
    params: MCMCParams,
    host_device_count: int | None,
    progress_bar: bool,
    progress_label: str | None = None,
) -> dict[str, object]:
    configure_parallel_chains(
        num_chains=params.num_chains,
        chain_method=params.chain_method,
        host_device_count=host_device_count,
    )

    data = simulation["data"]
    choices = jnp.asarray(data["choices"])
    source_choices = jnp.asarray(data["source_choices"])
    hidden_source = jnp.asarray(data["hidden_source"])
    true_params = SimParams.from_mapping(simulation["participant_parameters"])

    with labeled_progress_bar(progress_label if progress_bar else None):
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
    summary = summarize_posterior(posterior_frame(flat_samples), parameter_values(true_params))
    summary.index.name = "parameter"
    return {
        "chain_method": mcmc.chain_method,
        "diagnostics": mcmc_diagnostics(mcmc),
        "samples": extract_parameter_samples(chain_samples),
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    params = MCMCParams.from_args(args)
    posterior = fit_posterior(
        simulation=read_simulation_zarr(args.simulation),
        params=params,
        host_device_count=args.host_device_count,
        progress_bar=args.progress,
        progress_label=args.progress_label,
    )
    write_posterior_zarr(
        args.output,
        diagnostics=posterior["diagnostics"],
        samples=posterior["samples"],
        summary=posterior["summary"],
    )
    print(
        f"posterior {args.output}: "
        f"chain_method={posterior['chain_method']} samples={params.num_samples}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
