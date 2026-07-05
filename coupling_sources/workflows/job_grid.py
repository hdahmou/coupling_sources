from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from coupling_sources.core.parameters.simulation import SimParams
from coupling_sources.core.parameters.stimuli import StimuliParams


@dataclass(frozen=True)
class GridRun:
    run_id: str
    stimuli: StimuliParams
    simulation: SimParams


def _expand_product_config(block: dict[str, Any]) -> list[dict[str, Any]]:
    if "grid" not in block:
        return [block]

    grid = block["grid"]
    if grid["mode"] != "product":
        raise ValueError("grid.mode must be 'product'")

    base = {key: value for key, value in block.items() if key not in {"defaults", "grid"}}
    base.update(block.get("defaults", {}))
    axes = grid["axes"]
    names = tuple(axes)
    return [
        {**base, **{name: value for name, value in zip(names, values, strict=True)}}
        for values in product(*(axes[name] for name in names))
    ]


def _sample_sim_params_from_prior(seed: int, draws: int) -> list[SimParams]:
    samples = SimParams.sample_prior_draws(seed, draws)
    parameter_names = SimParams.names()
    return [
        SimParams(*(samples[name][index] for name in parameter_names))
        for index in range(draws)
    ]


def build_run_grid(config: dict[str, Any]) -> list[GridRun]:
    stimuli = [
        StimuliParams.from_mapping(values)
        for values in _expand_product_config(config["stimuli"])
    ]
    simulation = config["simulation"]
    if "participant_parameters" in simulation:
        parameters = [
            SimParams.from_mapping(values)
            for values in _expand_product_config(simulation["participant_parameters"])
        ]
    else:
        prior = config["prior"]
        parameters = _sample_sim_params_from_prior(
            seed=prior["seed"],
            draws=prior["parameter_draws"],
        )

    run_inputs = product(stimuli, parameters)
    return [
        GridRun(f"job_{index:06d}", stimuli, parameters)
        for index, (stimuli, parameters) in enumerate(run_inputs)
    ]


def _csv_value(value: object) -> object:
    if isinstance(value, list | tuple | dict):
        return json.dumps(value)
    return value


def write_job_grid(output: Path, runs: Sequence[GridRun]) -> None:
    fieldnames = ["job_id", "stimuli_path", "simulation_path", "posterior_path"]
    stimuli_names = StimuliParams.names()
    simulation_names = SimParams.names()
    fieldnames.extend(f"stimuli_{name}" for name in stimuli_names)
    fieldnames.extend(f"simulation_{name}" for name in simulation_names)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            job_path = Path("jobs") / run.run_id
            row = {
                "job_id": run.run_id,
                "stimuli_path": str(job_path / "stimuli.zarr"),
                "simulation_path": str(job_path / "simulation.zarr"),
                "posterior_path": str(job_path / "posterior.zarr"),
            }
            row.update(
                {
                    f"stimuli_{name}": _csv_value(getattr(run.stimuli, name))
                    for name in stimuli_names
                }
            )
            row.update(
                {
                    f"simulation_{name}": _csv_value(getattr(run.simulation, name))
                    for name in simulation_names
                }
            )
            writer.writerow(row)
