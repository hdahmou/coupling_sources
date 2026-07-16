from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from itertools import product
from pathlib import Path
from typing import Any

from coupling_sources.core.data import GainConstants, StimulusSpec
from coupling_sources.core.gain_calibration import (
    calibrate_gain_constants,
    gain_calibration_design,
    gain_calibration_key,
    gain_calibration_record,
    gain_calibration_spec,
)
from coupling_sources.core.priors import CouplingPriors


@dataclass(frozen=True)
class GridRun:
    run_id: str
    stimuli: StimulusSpec
    gains: GainConstants
    simulation: dict[str, float]
    gain_calibration: dict[str, object]


@dataclass(frozen=True)
class ResolvedGains:
    constants: GainConstants
    calibration: dict[str, object]


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
    return StimulusSpec.from_mapping(values)


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


def _gain_calibration_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    calibration = config.get("gain_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("gain_calibration must be an object")

    mode = calibration.get("mode")
    has_constants = "gain_constants" in config
    if mode == "fixed" and not has_constants:
        raise ValueError("fixed gain_calibration requires gain_constants")
    if mode == "auto" and has_constants:
        raise ValueError("automatic gain_calibration cannot include gain_constants")
    if mode not in {"auto", "fixed"}:
        raise ValueError("gain_calibration.mode must be 'auto' or 'fixed'")
    return calibration


def _monte_carlo_settings(calibration: Mapping[str, Any]) -> tuple[int, int]:
    try:
        seed = int(calibration["seed"])
        n_monte_carlo_replicates = int(
            calibration["n_monte_carlo_replicates"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "gain_calibration requires integer seed and "
            "n_monte_carlo_replicates"
        ) from error
    if n_monte_carlo_replicates < 2:
        raise ValueError(
            "gain_calibration.n_monte_carlo_replicates must be at least 2"
        )
    return seed, n_monte_carlo_replicates


def _fixed_gains(
    config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    stimuli: Sequence[StimulusSpec],
) -> ResolvedGains:
    _monte_carlo_settings(calibration)
    gains = GainConstants.from_mapping(config["gain_constants"])
    estimates = calibration.get("estimates")
    if not isinstance(estimates, Mapping):
        raise ValueError("fixed gain_calibration.estimates must be an object")
    for name in ("g_star_y", "g_star_z"):
        if getattr(gains, name) != round(float(estimates[name]), 3):
            raise ValueError(
                f"{name} does not match the rounded gain_calibration estimate"
            )
    for stimulus in stimuli:
        for name, expected in gain_calibration_design(stimulus).items():
            if calibration.get(name) != expected:
                raise ValueError(
                    f"fixed gain_calibration.{name} does not match the stimulus "
                    "design"
                )
    return ResolvedGains(gains, dict(calibration))


def _resolve_gains(
    config: Mapping[str, Any],
    stimuli: Sequence[StimulusSpec],
) -> list[ResolvedGains]:
    calibration_config = _gain_calibration_config(config)
    if calibration_config["mode"] == "fixed":
        resolved = _fixed_gains(config, calibration_config, stimuli)
        return [resolved for _ in stimuli]

    seed, n_monte_carlo_replicates = _monte_carlo_settings(calibration_config)
    cache: dict[tuple[object, ...], ResolvedGains] = {}
    resolved: list[ResolvedGains] = []
    for stimulus in stimuli:
        key = gain_calibration_key(stimulus)
        if key not in cache:
            calibration_spec = gain_calibration_spec(
                stimulus,
                seed=seed,
                n_monte_carlo_replicates=n_monte_carlo_replicates,
            )
            calibration = calibrate_gain_constants(calibration_spec)
            gains = GainConstants(
                g_star_y=calibration.g_star_y,
                g_star_z=calibration.g_star_z,
            )
            cache[key] = ResolvedGains(
                constants=gains,
                calibration=gain_calibration_record(
                    calibration_spec,
                    calibration,
                    mode="auto",
                ),
            )
        resolved.append(cache[key])
    return resolved


def build_run_grid(config: dict[str, Any]) -> list[GridRun]:
    stimuli = [
        _stimulus_parameters(values)
        for values in _expand_product_config(config["stimuli"])
    ]
    resolved_gains = _resolve_gains(config, stimuli)
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

    run_inputs = (
        (stimulus, gains, simulation_parameters)
        for stimulus, gains in zip(stimuli, resolved_gains, strict=True)
        for simulation_parameters in parameters
    )
    return [
        GridRun(
            f"job_{index:06d}",
            stimulus,
            gains.constants,
            simulation_parameters,
            gains.calibration,
        )
        for index, (stimulus, gains, simulation_parameters) in enumerate(run_inputs)
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
    fieldnames.extend(("gain_g_star_y", "gain_g_star_z"))
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
                    "gain_g_star_y": run.gains.g_star_y,
                    "gain_g_star_z": run.gains.g_star_z,
                }
            )
            row.update(
                {
                    f"simulation_{name}": _csv_value(run.simulation[name])
                    for name in simulation_names
                }
            )
            writer.writerow(row)


def write_gain_calibrations(output: Path, runs: Sequence[GridRun]) -> None:
    """Write each distinct resolved calibration once for run provenance."""
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for run in runs:
        record = {
            **run.gain_calibration,
            "gain_constants": asdict(run.gains),
        }
        serialized = json.dumps(record, sort_keys=True)
        if serialized not in seen:
            seen.add(serialized)
            records.append(record)

    output.write_text(json.dumps({"calibrations": records}, indent=2) + "\n")
