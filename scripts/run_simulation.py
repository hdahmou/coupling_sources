from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from coupling_sources.core.parameters.simulation import SimParams
from coupling_sources.core.simulation import generate_simulation
from coupling_sources.utils.artifacts import read_stimuli_zarr, write_simulation_zarr


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one simulation from stimuli.")
    parser.add_argument("--stimuli", type=Path, required=True, help="Stimuli Zarr directory.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--kappa-p", type=float, required=True)
    parser.add_argument("--c0", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True, help="Output Zarr directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    simulation = generate_simulation(
        seed=args.seed,
        params=SimParams(
            alpha=args.alpha,
            gamma=args.gamma,
            kappa_p=args.kappa_p,
            c0=args.c0,
        ),
        stimuli=read_stimuli_zarr(args.stimuli),
    )
    write_simulation_zarr(args.output, simulation)
    print(f"simulation: choices={np.asarray(simulation.choices).shape}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
