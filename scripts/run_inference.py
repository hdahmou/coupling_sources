from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from coupling_sources.core.data import GainConstants, MCMCSpecs, StimulusBatch
from coupling_sources.core.inference import run_inference
from coupling_sources.utils.artifacts import Artifacts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simulation and posterior inference for one stimulus artifact."
    )
    parser.add_argument(
        "--stimuli",
        type=Path,
        required=True,
        help="Stimuli .npz artifact.",
    )
    parser.add_argument("--simulation-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--gamma-y", type=float, required=True)
    parser.add_argument("--gamma-z", type=float, required=True)
    parser.add_argument("--g-star-y", type=float, required=True)
    parser.add_argument("--g-star-z", type=float, required=True)
    parser.add_argument("--vm-concentration", type=float, required=True)
    parser.add_argument("--c0", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-warmup", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--num-chains", type=int, required=True)
    parser.add_argument("--chain-method", type=str, required=True)
    parser.add_argument("--target-accept-prob", type=float, required=True)
    parser.add_argument("--max-tree-depth", type=int, required=True)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-label", default=None)
    parser.add_argument(
        "--simulation-output",
        type=Path,
        default=None,
        help="Optional simulation .npz artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Posterior .npz artifact.",
    )
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
        stimuli=StimulusBatch.from_config(Artifacts.read_config(args.stimuli)),
        simulation_parameters={
            "learningRate": args.learning_rate,
            "gamma_y": args.gamma_y,
            "gamma_z": args.gamma_z,
            "vmConcentration": args.vm_concentration,
            "c0": args.c0,
        },
        simulation_seed=args.simulation_seed,
        gains=GainConstants(
            g_star_y=args.g_star_y,
            g_star_z=args.g_star_z,
        ),
        mcmc=mcmc,
        progress_bar=args.progress,
        progress_label=args.progress_label,
    )
    if args.simulation_output is not None:
        Artifacts.write_config(
            args.simulation_output,
            result["simulation"].to_config(),
            float_dtype=np.float16,
        )
    Artifacts.write_dist_samples(args.output, result["samples"])
    Artifacts.write_mcmc_diagnostics(args.output, result["diagnostics"])
    print(
        f"posterior {args.output}: chain_method={result['chain_method']} "
        f"samples={mcmc.num_samples}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
