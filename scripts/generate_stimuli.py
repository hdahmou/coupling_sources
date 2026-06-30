from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import jax

from artifacts import array, write_stimuli_zarr
from coupling_sources.parameters.stimuli import StimuliParams
from coupling_sources.stimuli import generate_stimuli


def generate_stimulus(params: StimuliParams) -> dict[str, object]:
    batch = generate_stimuli(
        jax.random.PRNGKey(params.seed),
        n_rounds=params.n_rounds,
        n_trials_full=params.n_trials_full,
        n_trials_partial=params.n_trials_partial,
        n_sources=params.n_sources,
        sigma=params.sigma,
        lam=params.lam,
        coupled=params.coupled_sources,
    )
    return {
        "truth": batch.truth,
        "source_choices": batch.source_choices,
        "hidden_source": batch.hidden_source,
        "complete_source_choices": batch.complete_source_choices,
        "covariance": batch.covariance,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stimuli.")
    StimuliParams.add_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="Output Zarr directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stimuli = generate_stimulus(StimuliParams.from_args(args))
    write_stimuli_zarr(args.output, stimuli)
    print(
        f"stimuli: source_choices={array(stimuli['source_choices']).shape}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
