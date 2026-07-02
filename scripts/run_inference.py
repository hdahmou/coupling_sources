from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from coupling_sources.core.inference import fit_posterior
from coupling_sources.core.parameters.mcmc import MCMCParams
from coupling_sources.utils.artifacts import read_simulation_zarr, write_posterior_zarr


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run posterior inference for one simulation.")
    parser.add_argument("--simulation", type=Path, required=True, help="Simulation Zarr directory.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-warmup", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--num-chains", type=int, required=True)
    parser.add_argument("--chain-method", type=str, required=True)
    parser.add_argument("--target-accept-prob", type=float, required=True)
    parser.add_argument("--max-tree-depth", type=int, required=True)
    parser.add_argument("--host-device-count", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-label", default=None)
    parser.add_argument("--output", type=Path, required=True, help="Output Zarr directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    params = MCMCParams(
        seed=args.seed,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method=args.chain_method,
        target_accept_prob=args.target_accept_prob,
        max_tree_depth=args.max_tree_depth,
    )
    posterior = fit_posterior(
        simulation=read_simulation_zarr(args.simulation),
        params=params,
        host_device_count=args.host_device_count,
        progress_bar=args.progress,
        progress_label=args.progress_label,
    )
    write_posterior_zarr(
        args.output,
        diagnostics=posterior["diagnostics"],
        samples=posterior["samples"],
    )
    print(
        f"posterior {args.output}: "
        f"chain_method={posterior['chain_method']} samples={params.num_samples}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
