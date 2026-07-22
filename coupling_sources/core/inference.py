from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

import jax
import numpyro

from coupling_sources.core.data import (
    FitData,
    GainConstants,
    MCMCSpecs,
    SimulationBatch,
    StimulusBatch,
    prepare_simulation_for_fit,
    prepare_stimuli_for_fit,
)
from coupling_sources.core.diagnostics import mcmc_diagnostics
from coupling_sources.core.model import model_coupling
from coupling_sources.core.priors import CouplingPriors
from coupling_sources.core.simulator import simulate_coupling
from coupling_sources.utils.progress import labeled_progress_bar


class FitResult(TypedDict):
    chain_method: str
    diagnostics: dict[str, object]
    samples: dict[str, Any]


class InferenceResult(FitResult):
    simulation: SimulationBatch


def run_inference(
    *,
    stimuli: StimulusBatch,
    simulation_parameters: Mapping[str, Any],
    simulation_seed: int,
    gains: GainConstants,
    mcmc: MCMCSpecs,
    progress_bar: bool,
    progress_label: str | None = None,
) -> InferenceResult:
    """Run the full pipeline: stimuli -> simulator -> NumPyro MCMC."""
    simulation = simulate_for_inference(
        stimuli=stimuli,
        simulation_parameters=simulation_parameters,
        simulation_seed=simulation_seed,
        gains=gains,
    )
    fit_result = fit_model(
        fit_data=prepare_simulation_for_fit(simulation),
        c0=simulation_parameters["c0"],
        gains=gains,
        mcmc=mcmc,
        progress_bar=progress_bar,
        progress_label=progress_label,
    )
    return {"simulation": simulation, **fit_result}


def simulate_for_inference(
    *,
    stimuli: StimulusBatch,
    simulation_parameters: Mapping[str, Any],
    simulation_seed: int,
    gains: GainConstants,
) -> SimulationBatch:
    numpyro.enable_x64(True)
    stimuli_for_fit = prepare_stimuli_for_fit(stimuli)
    return simulate_coupling(
        seed=simulation_seed,
        batch=stimuli_for_fit,
        parameters=simulation_parameters,
        gains=gains,
    )


def fit_model(
    *,
    fit_data: FitData,
    c0: float | jax.Array,
    gains: GainConstants,
    mcmc: MCMCSpecs,
    progress_bar: bool,
    progress_label: str | None = None,
) -> FitResult:
    numpyro.enable_x64(True)

    kernel = numpyro.infer.NUTS(
        model_coupling,
        target_accept_prob=mcmc.target_accept_prob,
        max_tree_depth=mcmc.max_tree_depth,
    )

    def postprocess_fit_samples(latent_samples: dict[str, Any]) -> dict[str, Any]:
        samples = kernel.postprocess_fn((fit_data, c0, gains), {})(latent_samples)
        return {
            name: value for name, value in samples.items() if not name.startswith("~")
        }

    mcmc_runner = numpyro.infer.MCMC(
        kernel,
        num_warmup=mcmc.num_warmup,
        num_samples=mcmc.num_samples,
        num_chains=mcmc.num_chains,
        chain_method=mcmc.chain_method,
        progress_bar=progress_bar,
        postprocess_fn=postprocess_fit_samples,
    )
    with labeled_progress_bar(progress_label if progress_bar else None):
        mcmc_runner.run(jax.random.PRNGKey(mcmc.seed), fit_data, c0, gains)

    chain_samples = mcmc_runner.get_samples(group_by_chain=True)
    parameter_names = CouplingPriors.parameter_sites()
    return {
        "chain_method": mcmc.chain_method,
        "diagnostics": mcmc_diagnostics(mcmc_runner),
        "samples": {
            name: jax.device_get(chain_samples[name]) for name in parameter_names
        },
    }
