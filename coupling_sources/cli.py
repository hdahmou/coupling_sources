from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

from coupling_sources.workflows.parameter_recovery import run_parameter_recovery


def parse_coordinator_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create posterior artifact trees on a Dask cluster."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compute-config", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=None)
    return parser.parse_args(argv)


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def coordinator_main(argv: Sequence[str] | None = None) -> int:
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        args = parse_coordinator_args(argv)
        return run_parameter_recovery(
            config_path=args.config,
            compute_config_path=args.compute_config,
            artifacts_root=args.artifacts_root,
        )
    except KeyboardInterrupt:
        print(
            "parameter recovery interrupted; cancelling outstanding jobs",
            file=sys.stderr,
        )
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
