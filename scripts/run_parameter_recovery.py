from __future__ import annotations

import sys
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from collections.abc import Sequence
from pathlib import Path

from coupling_sources.workflows.parameter_recovery import run_parameter_recovery


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create posterior artifact trees from a config file.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compute-config", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_parameter_recovery(
        config_path=args.config,
        compute_config_path=args.compute_config,
        artifacts_root=args.artifacts_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
