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

from coupling_sources.core.data import MCMCSpecs
from coupling_sources.core.inference import run_inference
from coupling_sources.utils.artifacts import read_stimuli_zarr, write_posterior_zarr, write_simulation_zarr


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulation and posterior inference for one stimulus artifact.")
    parser.add_argument("--stimuli", type=Path, required=True, help="Stimuli Zarr directory.")
    parser.add_argument("--simulation-seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gain-y", type=float, required=True)
    parser.add_argument("--gain-z", type=float, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-warmup", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--num-chains", type=int, required=True)
    parser.add_argument("--chain-method", type=str, required=True)
    parser.add_argument("--target-accept-prob", type=float, required=True)
    parser.add_argument("--max-tree-depth", type=int, required=True)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-label", default=None)
    parser.add_argument("--simulation-output", type=Path, default=None, help="Optional simulation Zarr output.")
    parser.add_argument("--output", type=Path, required=True, help="Posterior Zarr output directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mcmc = MCMCSpecs(
        seed=args.seed,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method=args.chain_method,
        target_accept_prob=args.target_accept_prob,
        max_tree_depth=args.max_tree_depth,
    )
    result = run_inference(
        stimuli=read_stimuli_zarr(args.stimuli),
        simulation_parameters={
            "alpha": args.alpha,
            "gain_y": args.gain_y,
            "gain_z": args.gain_z,
            "kappa": args.kappa,
        },
        simulation_seed=args.simulation_seed,
        mcmc=mcmc,
        progress_bar=args.progress,
        progress_label=args.progress_label,
    )
    if args.simulation_output is not None:
        write_simulation_zarr(args.simulation_output, result["simulation"])
    write_posterior_zarr(
        args.output,
        diagnostics=result["diagnostics"],
        samples=result["samples"],
    )
    print(
        f"posterior {args.output}: chain_method={result['chain_method']} samples={mcmc.num_samples}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
