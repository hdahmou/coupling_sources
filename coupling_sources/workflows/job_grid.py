from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from itertools import product
from pathlib import Path
from typing import Any

from coupling_sources.core.data import StimulusSpec
from coupling_sources.core.priors import CouplingPriors


@dataclass(frozen=True)
class GridRun:
    run_id: str
    stimuli: StimulusSpec
    simulation: dict[str, float]


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


def _stimulus_parameters(values: Mapping[str, Any]) -> StimulusSpec:
    if "mean_angle_difference_band" in values:
        raise ValueError("mean_angle_difference_band was removed; use resultant_floor")

    params = {item.name: values[item.name] for item in fields(StimulusSpec)}
    params["coupled_sources"] = tuple(params["coupled_sources"])
    return StimulusSpec(**params)


def _sample_sim_params_from_prior(
    seed: int,
    draws: int,
    c0: float,
) -> list[dict[str, float]]:
    samples = CouplingPriors.sample_parameter_draws(seed, draws, c0)
    parameter_names = CouplingPriors.simulation_parameter_sites()
    return [
        {name: float(samples[name][index]) for name in parameter_names}
        for index in range(draws)
    ]


def build_run_grid(config: dict[str, Any]) -> list[GridRun]:
    stimuli = [
        _stimulus_parameters(values)
        for values in _expand_product_config(config["stimuli"])
    ]
    simulation_config = config["simulation"]
    c0 = float(simulation_config["c0"])
    if "participant_parameters" in simulation_config:
        parameters = [
            CouplingPriors.parameters_as_floats({**values, "c0": c0})
            for values in _expand_product_config(simulation_config["participant_parameters"])
        ]
    else:
        prior = config["prior"]
        parameters = _sample_sim_params_from_prior(
            seed=prior["seed"],
            draws=prior["parameter_draws"],
            c0=c0,
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
    stimuli_names = tuple(item.name for item in fields(StimulusSpec))
    simulation_names = CouplingPriors.simulation_parameter_sites()
    fieldnames.extend(f"stimuli_{name}" for name in stimuli_names)
    fieldnames.extend(f"simulation_{name}" for name in simulation_names)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            job_path = Path("jobs") / run.run_id
            row: dict[str, object] = {
                "job_id": run.run_id,
                "stimuli_path": str(job_path / "stimuli.npz"),
                "simulation_path": str(job_path / "simulation.npz"),
                "posterior_path": str(job_path / "posterior.npz"),
            }
            row.update(
                {
                    f"stimuli_{name}": _csv_value(getattr(run.stimuli, name))
                    for name in stimuli_names
                }
            )
            row.update(
                {
                    f"simulation_{name}": _csv_value(run.simulation[name])
                    for name in simulation_names
                }
            )
            writer.writerow(row)
