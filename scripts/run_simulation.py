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

from coupling_sources.core.data import StimulusBatch
from coupling_sources.core.simulator import simulate_coupling
from coupling_sources.utils.artifacts import Artifacts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one simulation from stimuli.")
    parser.add_argument("--stimuli", type=Path, required=True, help="Stimuli .npz artifact.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--g-y", type=float, required=True)
    parser.add_argument("--g-z", type=float, required=True)
    parser.add_argument("--vm-concentration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Simulation .npz artifact.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stimuli = StimulusBatch.from_config(Artifacts.read_config(args.stimuli))
    simulation = simulate_coupling(
        seed=args.seed,
        batch=stimuli,
        parameters={
            "learningRate": args.learning_rate,
            "g_y": args.g_y,
            "g_z": args.g_z,
            "vmConcentration": args.vm_concentration,
        },
    )
    Artifacts.write_config(args.output, simulation.to_config(), float_dtype=np.float16)
    responses = simulation.responses
    print(f"simulation: choices={np.asarray(responses).shape}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
