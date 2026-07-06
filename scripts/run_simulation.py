from __future__ import annotations

import sys
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from coupling_sources.core.simulator import simulate_coupling
from coupling_sources.utils.artifacts import read_stimuli_zarr, write_simulation_zarr


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one simulation from stimuli.")
    parser.add_argument("--stimuli", type=Path, required=True, help="Stimuli Zarr directory.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta-y", type=float, required=True)
    parser.add_argument("--beta-z", type=float, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output Zarr directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stimuli = read_stimuli_zarr(args.stimuli)
    simulation = simulate_coupling(
        seed=args.seed,
        batch=stimuli,
        parameters={
            "alpha": args.alpha,
            "beta_y": args.beta_y,
            "beta_z": args.beta_z,
            "kappa": args.kappa,
        },
    )
    write_simulation_zarr(args.output, simulation)
    responses = simulation.responses
    print(f"simulation: choices={np.asarray(responses).shape}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
