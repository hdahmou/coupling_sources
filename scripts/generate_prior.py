from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from artifacts import write_prior_zarr
from coupling_sources.parameter_prior import sample_parameter_prior


def generate_prior(seed: int, draws: int) -> dict[str, object]:
    return sample_parameter_prior(seed, draws)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prior samples.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output Zarr directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prior_samples = generate_prior(args.seed, args.draws)
    write_prior_zarr(args.output, prior_samples)
    print(f"prior draws: {args.draws}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
