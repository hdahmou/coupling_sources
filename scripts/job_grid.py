from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from coupling_sources.parameter_prior import sample_prior_parameter_sets
from coupling_sources.parameters.simulation import SimParams
from coupling_sources.parameters.stimuli import StimuliParams


@dataclass(frozen=True)
class GridRun:
    run_id: str
    stimuli: StimuliParams
    simulation: SimParams

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "stimuli": self.stimuli.as_dict(),
            "simulation": self.simulation.as_dict(),
        }


def expand_grid(block: dict[str, Any]) -> list[dict[str, Any]]:
    if "grid" not in block:
        return [block]

    grid = block["grid"]
    if grid["mode"] != "product":
        raise ValueError("grid.mode must be 'product'")

    base = {key: value for key, value in block.items() if key not in {"defaults", "grid"}}
    base.update(block.get("defaults", {}))
    axes = grid["axes"]
    names = list(axes)
    return [
        {**base, **dict(zip(names, values, strict=True))}
        for values in product(*(axes[name] for name in names))
    ]


def stimulus_grid(config: dict[str, Any]) -> list[StimuliParams]:
    return [StimuliParams.from_mapping(values) for values in expand_grid(config["stimuli"])]


def parameter_grid(config: dict[str, Any]) -> list[SimParams]:
    simulation = config["simulation"]
    if "participant_parameters" in simulation:
        return [
            SimParams.from_mapping(values)
            for values in expand_grid(simulation["participant_parameters"])
        ]

    prior = config["prior"]
    return sample_prior_parameter_sets(seed=prior["seed"], draws=prior["parameter_draws"])


def build_run_grid(config: dict[str, Any]) -> list[GridRun]:
    run_inputs = product(stimulus_grid(config), parameter_grid(config))
    return [
        GridRun(f"job_{index:06d}", stimuli, parameters)
        for index, (stimuli, parameters) in enumerate(run_inputs)
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand a config into concrete stimulus x simulator runs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--count", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs = build_run_grid(json.loads(args.config.read_text()))
    if args.count:
        print(len(runs))
    else:
        print(json.dumps([run.as_dict() for run in runs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
