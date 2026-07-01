from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from coupling_sources.core.parameters.stimuli import StimuliParams
from coupling_sources.core.stimuli import generate_stimulus
from coupling_sources.utils.artifacts import write_stimuli_zarr


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one stimulus artifact.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-sources", type=int, required=True)
    parser.add_argument("--n-rounds", type=int, required=True)
    parser.add_argument("--n-trials-full", type=int, required=True)
    parser.add_argument("--n-trials-partial", type=int, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--coupled-sources", type=int, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output Zarr directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stimuli = generate_stimulus(
        StimuliParams(
            seed=args.seed,
            n_sources=args.n_sources,
            n_rounds=args.n_rounds,
            n_trials_full=args.n_trials_full,
            n_trials_partial=args.n_trials_partial,
            sigma=args.sigma,
            lam=args.lam,
            coupled_sources=tuple(args.coupled_sources),
        )
    )
    write_stimuli_zarr(args.output, stimuli)
    print(
        f"stimuli: source_choices={np.asarray(stimuli.source_choices).shape}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
