from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
import zarr

from coupling_sources.core.parameters.simulation import SimParams
from coupling_sources.core.simulation import SimulationArtifact, SimulationTrace
from coupling_sources.core.stimuli import StimulusBatch

STIMULI_ARRAYS = (
    "truth",
    "source_choices",
    "hidden_source",
    "complete_source_choices",
    "covariance",
)
SIMULATION_ARRAYS = ("choices", "source_choices", "hidden_source")


def _array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _write_arrays(group: Any, arrays: Mapping[str, Any]) -> None:
    for name, value in arrays.items():
        group.create_array(name, data=_array(value))


def write_prior_zarr(output: Path, prior_samples: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(output, mode="w")
    _write_arrays(root.create_group("prior_samples"), prior_samples)


def write_stimuli_zarr(output: Path, stimuli: StimulusBatch) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(output, mode="w")
    _write_arrays(root, {name: getattr(stimuli, name) for name in STIMULI_ARRAYS})


def read_stimuli_zarr(path: Path) -> StimulusBatch:
    root = zarr.open_group(path, mode="r")
    return StimulusBatch(**{name: np.asarray(root[name]) for name in STIMULI_ARRAYS})


def write_simulation_zarr(output: Path, simulation: SimulationArtifact) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(output, mode="w")
    _write_arrays(root, {name: getattr(simulation, name) for name in SIMULATION_ARRAYS})
    _write_arrays(
        root.create_group("participant_parameters"),
        simulation.participant_parameters.as_dict(),
    )
    _write_arrays(
        root.create_group("trace"),
        {field.name: getattr(simulation.trace, field.name) for field in fields(SimulationTrace)},
    )


def read_simulation_zarr(path: Path) -> SimulationArtifact:
    root = zarr.open_group(path, mode="r")
    parameters = root["participant_parameters"]
    trace = root["trace"]
    return SimulationArtifact(
        choices=np.asarray(root["choices"]),
        source_choices=np.asarray(root["source_choices"]),
        hidden_source=np.asarray(root["hidden_source"]),
        participant_parameters=SimParams.from_mapping(
            {name: np.asarray(parameters[name]) for name in SimParams.names()}
        ),
        trace=SimulationTrace(
            **{
                field.name: np.asarray(trace[field.name])
                for field in fields(SimulationTrace)
            }
        ),
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_posterior_zarr(
    output: Path,
    *,
    diagnostics: dict[str, Any],
    samples: dict[str, Any],
    summary: pd.DataFrame,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(output, mode="w")
    root.attrs["diagnostics"] = _json_ready(diagnostics)

    samples_group = root.create_group("posterior_samples")
    samples_group.attrs["dims"] = ["chain", "draw"]
    for parameter, values in samples.items():
        data = _array(values)
        chunks = (1, data.shape[1]) if data.ndim == 2 else data.shape
        samples_group.create_array(parameter, data=data, chunks=chunks)

    summary_group = root.create_group("posterior_summary")
    summary_group.attrs["parameter"] = list(summary.index)
    summary_group.attrs["dims"] = ["parameter"]
    for column in summary.columns:
        summary_group.create_array(column, data=summary[column].to_numpy())
