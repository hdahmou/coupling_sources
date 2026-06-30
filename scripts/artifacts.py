from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
import zarr

from coupling_sources.parameters.simulation import SimParams

STIMULI_ARRAYS = (
    "truth",
    "source_choices",
    "hidden_source",
    "complete_source_choices",
    "covariance",
)
SIMULATION_ARRAYS = ("choices", "source_choices", "hidden_source")


def array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def write_arrays(group: Any, arrays: dict[str, Any]) -> None:
    for name, value in arrays.items():
        group.create_array(name, data=array(value))


def write_prior_zarr(output: Path, prior_samples: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(output, mode="w")
    write_arrays(root.create_group("prior_samples"), prior_samples)


def write_stimuli_zarr(output: Path, stimuli: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(output, mode="w")
    write_arrays(root, stimuli)


def read_stimuli_zarr(path: Path) -> dict[str, np.ndarray]:
    root = zarr.open_group(path, mode="r")
    return {name: np.asarray(root[name]) for name in STIMULI_ARRAYS}


def write_simulation_zarr(output: Path, simulation: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(output, mode="w")
    write_arrays(root, {name: simulation[name] for name in SIMULATION_ARRAYS})
    write_arrays(root.create_group("participant_parameters"), simulation["participant_parameters"])
    write_arrays(root.create_group("trace"), simulation["trace"])


def read_simulation_zarr(path: Path) -> dict[str, Any]:
    root = zarr.open_group(path, mode="r")
    parameters = root["participant_parameters"]
    return {
        "participant_parameters": {
            field.name: np.asarray(parameters[field.name])
            for field in fields(SimParams)
        },
        "data": {name: np.asarray(root[name]) for name in SIMULATION_ARRAYS},
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
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
    root.attrs["diagnostics"] = json_ready(diagnostics)

    samples_group = root.create_group("posterior_samples")
    samples_group.attrs["dims"] = ["chain", "draw"]
    for parameter, values in samples.items():
        data = array(values)
        chunks = (1, data.shape[1]) if data.ndim == 2 else data.shape
        samples_group.create_array(parameter, data=data, chunks=chunks)

    summary_group = root.create_group("posterior_summary")
    summary_group.attrs["parameter"] = list(summary.index)
    summary_group.attrs["dims"] = ["parameter"]
    for column in summary.columns:
        summary_group.create_array(column, data=summary[column].to_numpy())
