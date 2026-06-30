from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp

from artifacts import array, read_stimuli_zarr, write_simulation_zarr
from coupling_sources.parameters.simulation import SimParams
from coupling_sources.simulator import simulate_choices


def generate_simulation(
    *,
    seed: int,
    params: SimParams,
    stimuli: dict[str, object],
) -> dict[str, object]:
    source_choices = jnp.asarray(stimuli["source_choices"])
    if source_choices.ndim != 3:
        raise ValueError(f"source_choices must be 3D, got {tuple(source_choices.shape)}")
    hidden_source = jnp.asarray(stimuli["hidden_source"])
    result = simulate_choices(jax.random.PRNGKey(seed), params, source_choices, hidden_source)
    return {
        "choices": result.choices,
        "source_choices": source_choices,
        "hidden_source": hidden_source,
        "participant_parameters": params.as_dict(),
        "trace": {
            "response_mean": result.trace.response_mean,
            "truth_mean": result.trace.truth_mean,
            "hidden_mean": result.trace.hidden_mean,
            "source_weights": result.trace.source_weights,
            "pre_update_agreement_scores": result.trace.pre_update_agreement_scores,
            "choice_logpdf": result.trace.choice_logpdf,
            "is_probe": result.trace.is_probe,
            "hidden_source": result.trace.hidden_source,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simulation from stimuli.")
    parser.add_argument("--stimuli", type=Path, required=True, help="Stimuli Zarr directory.")
    parser.add_argument("--seed", type=int, required=True)
    SimParams.add_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="Output Zarr directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    simulation = generate_simulation(
        seed=args.seed,
        params=SimParams.from_args(args),
        stimuli=read_stimuli_zarr(args.stimuli),
    )
    write_simulation_zarr(args.output, simulation)
    print(f"simulation: choices={array(simulation['choices']).shape}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
