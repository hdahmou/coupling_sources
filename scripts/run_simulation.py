from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from coupling_sources.core.data import GainConstants, StimulusBatch
from coupling_sources.core.simulator import simulate_coupling
from coupling_sources.utils.artifacts import Artifacts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one simulation from stimuli.")
    parser.add_argument(
        "--stimuli",
        type=Path,
        required=True,
        help="Stimuli .npz artifact.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--gamma-y", type=float, required=True)
    parser.add_argument("--gamma-z", type=float, required=True)
    parser.add_argument("--g-star-y", type=float, required=True)
    parser.add_argument("--g-star-z", type=float, required=True)
    parser.add_argument("--vm-concentration", type=float, required=True)
    parser.add_argument("--c0", type=float, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Simulation .npz artifact.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stimuli = StimulusBatch.from_config(Artifacts.read_config(args.stimuli))
    simulation = simulate_coupling(
        seed=args.seed,
        batch=stimuli,
        parameters={
            "learningRate": args.learning_rate,
            "gamma_y": args.gamma_y,
            "gamma_z": args.gamma_z,
            "vmConcentration": args.vm_concentration,
            "c0": args.c0,
        },
        gains=GainConstants(
            g_star_y=args.g_star_y,
            g_star_z=args.g_star_z,
        ),
    )
    Artifacts.write_config(args.output, simulation.to_config(), float_dtype=np.float16)
    print(
        f"simulation: responses={np.asarray(simulation.responses).shape}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
