from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.stats import vonmises

from coupling_sources.utils.angles import centered_angle
from coupling_sources.core.parameters.simulation import SimParams
from coupling_sources.core.response import compute_response_trace
from coupling_sources.core.stimuli import StimulusBatch


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


@dataclass(frozen=True)
class SimulationArtifact:
    choices: jax.Array
    source_choices: jax.Array
    hidden_source: jax.Array
    participant_parameters: SimParams
    trace: SimulationTrace


def simulate_choices(
    key: jax.Array,
    params: SimParams,
    source_choices: jax.Array,
    hidden_source: jax.Array,
) -> SimulationResult:
    source_choices = jnp.asarray(source_choices)
    hidden_source = jnp.asarray(hidden_source)
    if source_choices.ndim != 3:
        raise ValueError("source_choices must have shape (source, round, time)")
    if hidden_source.shape != source_choices.shape[1:]:
        raise ValueError("hidden_source must have shape (round, time)")
    flow = compute_response_trace(params, source_choices, hidden_source)
    choices = (
        dist.VonMises(centered_angle(flow.response_mean_angle), params.kappa_p).sample(key)
        % (2 * jnp.pi)
    )

    trace = SimulationTrace(
        response_mean=flow.response_mean_angle,
        truth_mean=flow.truth_branch_mean_angle,
        hidden_mean=flow.hidden_source_branch_mean_angle,
        source_weights=flow.source_weights,
        pre_update_agreement_scores=flow.pre_update_agreement_scores,
        choice_logpdf=vonmises.logpdf(
            centered_angle(choices - flow.response_mean_angle), params.kappa_p
        ),
        is_probe=flow.is_probe_trial,
        hidden_source=hidden_source,
    )
    return SimulationResult(choices=choices, trace=trace)


def generate_simulation(
    *,
    seed: int,
    params: SimParams,
    stimuli: StimulusBatch,
) -> SimulationArtifact:
    source_choices = jnp.asarray(stimuli.source_choices)
    hidden_source = jnp.asarray(stimuli.hidden_source)
    result = simulate_choices(jax.random.PRNGKey(seed), params, source_choices, hidden_source)
    return SimulationArtifact(
        choices=result.choices,
        source_choices=source_choices,
        hidden_source=hidden_source,
        participant_parameters=params,
        trace=result.trace,
    )
