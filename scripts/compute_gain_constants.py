"""Compute the locked gain design constants g*_y and g*_z.

The fitted use parameters gamma_y and gamma_z are expressed in benchmark units
through fixed design constants. The constants are calibrated from the expected
relative-agreement signals under the screened production generator and exact
truth/probe trial mixture.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coupling_sources.core.data import StimulusSpec
from coupling_sources.core.gain_calibration import (
    DEFAULT_GAIN_CALIBRATION_REPLICATES,
    DEFAULT_GAIN_CALIBRATION_SEED,
    calibrate_gain_constants,
    gain_calibration_record,
    gain_calibration_stability,
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--resultant-floor-full", type=float, required=True)
    parser.add_argument(
        "--resultant-floor-leave-one-out",
        type=float,
        required=True,
    )
    parser.add_argument("--n-trials-full", type=int, required=True)
    parser.add_argument("--n-trials-partial", type=int, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_GAIN_CALIBRATION_SEED,
        help="locked Monte Carlo seed (record it)",
    )
    parser.add_argument(
        "--n-monte-carlo-replicates",
        type=int,
        default=DEFAULT_GAIN_CALIBRATION_REPLICATES,
        help="number of independently generated trial-list replicates",
    )
    parser.add_argument(
        "--max-relative-mc-se",
        type=_positive_float,
        default=None,
        help=(
            "optional maximum relative Monte Carlo SE for both gains; "
            "for example, 0.01 requires at most 1%%"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = StimulusSpec(
        seed=args.seed,
        n_participants=args.n_monte_carlo_replicates,
        n_sources=4,
        n_rounds=1,
        n_trials_full=args.n_trials_full,
        n_trials_partial=args.n_trials_partial,
        sigma=args.sigma,
        lam=args.lam,
        coupled_sources=(0, 1),
        resultant_floor_full=args.resultant_floor_full,
        resultant_floor_leave_one_out=args.resultant_floor_leave_one_out,
    )
    calibration = calibrate_gain_constants(spec)
    stability = gain_calibration_stability(calibration)

    print(
        f"design: sigma={spec.sigma}, lam={spec.lam}, "
        f"full floor={spec.resultant_floor_full}, "
        f"leave-one-out floor={spec.resultant_floor_leave_one_out}"
    )
    print(
        f"Monte Carlo: {spec.n_participants} independent trial-list replicates x "
        f"({spec.n_trials_full} truth + {spec.n_trials_partial} probe) trials, "
        f"seed={spec.seed}"
    )
    for name in (
        "s_coupled",
        "s_mixed",
        "s_independent",
        "delta_y",
        "delta_z",
    ):
        print(
            f"{name} = {getattr(calibration, name):.6f} "
            f"(MC se {getattr(calibration, f'{name}_se'):.6f})"
        )
    print()
    for name in ("g_star_y", "g_star_z"):
        interval = getattr(stability, f"{name}_mc_interval_95")
        print(
            f"{name} = {getattr(calibration, name):.3f} "
            f"(MC se {getattr(calibration, f'{name}_se'):.3f}; "
            f"relative MC se "
            f"{getattr(stability, f'{name}_relative_mc_se'):.2%}; "
            f"approx. 95% MC interval [{interval[0]:.3f}, {interval[1]:.3f}])"
        )
    max_relative_mc_se = stability.max_relative_mc_se
    if args.max_relative_mc_se is None:
        print(
            "stability: maximum relative MC se "
            f"{max_relative_mc_se:.2%} (no threshold requested)"
        )
        stable = True
    else:
        stable = max_relative_mc_se <= args.max_relative_mc_se
        status = "PASS" if stable else "FAIL"
        print(
            f"stability: {status}; maximum relative MC se "
            f"{max_relative_mc_se:.2%}, threshold {args.max_relative_mc_se:.2%}"
        )
    print()
    print(
        json.dumps(
            {
                "gain_constants": {
                    "g_star_y": round(calibration.g_star_y, 3),
                    "g_star_z": round(calibration.g_star_z, 3),
                },
                "gain_calibration": gain_calibration_record(
                    spec,
                    calibration,
                    mode="fixed",
                ),
            },
            indent=2,
        )
    )
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
