"""Zarr artifact serialization for coupling model data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import jax
import numpy as np
import zarr

from coupling_sources.core.data import (
    CouplingDiagnostics,
    SimulationBatch,
    StimulusBatch,
    StimulusMetadata,
)
from coupling_sources.core.priors import coupling_parameters_as_floats

T = TypeVar("T")
Tree = dict[str, Any]


# Zarr tree I/O


def _array_for_zarr(value: Any, *, float_dtype: Any = None) -> np.ndarray:
    data = np.asarray(jax.device_get(value))
    if float_dtype is not None and np.issubdtype(data.dtype, np.floating):
        return data.astype(float_dtype, copy=False)
    return data


def _write_zarr_tree(
    group: Any,
    tree: Mapping[str, Any],
    *,
    float_dtype: Any = None,
) -> None:
    for name, value in tree.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            _write_zarr_tree(
                group.create_group(name),
                value,
                float_dtype=float_dtype,
            )
            continue
        group.create_array(name, data=_array_for_zarr(value, float_dtype=float_dtype))


def _read_zarr_tree(group: Any) -> Tree:
    tree: Tree = {}
    for name in group.keys():
        node = group[name]
        tree[name] = (
            _read_zarr_tree(node) if hasattr(node, "keys") else np.asarray(node)
        )
    return tree


# Domain objects <-> storage trees


def _dataclass_to_tree(instance: object) -> Tree:
    return {item.name: getattr(instance, item.name) for item in fields(instance)}


def _dataclass_from_tree(cls: type[T], tree: Mapping[str, Any]) -> T:
    return cls(**{item.name: tree[item.name] for item in fields(cls)})


def _mapping_to_tree(value: CouplingDiagnostics | Mapping[str, Any]) -> Tree:
    if is_dataclass(value):
        return _dataclass_to_tree(value)
    return dict(value)


def _stimuli_to_tree(stimuli: StimulusBatch) -> Tree:
    tree = _dataclass_to_tree(stimuli)
    tree["metadata"] = _dataclass_to_tree(stimuli.metadata)
    return tree


def _stimuli_from_tree(tree: Mapping[str, Any]) -> StimulusBatch:
    values = dict(tree)
    values["metadata"] = _dataclass_from_tree(StimulusMetadata, tree["metadata"])
    return StimulusBatch(**values)


def _simulation_to_tree(simulation: SimulationBatch) -> Tree:
    if simulation.participant_parameters is None:
        raise KeyError("simulation is missing participant_parameters")
    if not simulation.diagnostics:
        raise KeyError("simulation is missing diagnostics")

    tree = _dataclass_to_tree(simulation)
    tree["stimuli"] = _stimuli_to_tree(simulation.stimuli)
    tree["participant_parameters"] = coupling_parameters_as_floats(
        simulation.participant_parameters
    )
    tree["diagnostics"] = _mapping_to_tree(simulation.diagnostics)
    return tree


def _simulation_from_tree(tree: Mapping[str, Any]) -> SimulationBatch:
    values = dict(tree)
    values["stimuli"] = _stimuli_from_tree(tree["stimuli"])
    values["participant_parameters"] = coupling_parameters_as_floats(
        tree["participant_parameters"]
    )
    values["diagnostics"] = _dataclass_from_tree(
        CouplingDiagnostics,
        tree["diagnostics"],
    )
    return SimulationBatch(**values)


# Stimulus, simulation, and prior artifacts


def write_prior_zarr(output: Path, prior_samples: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(output, mode="w")
    _write_zarr_tree(root, {"prior_samples": prior_samples}, float_dtype=np.float16)


def read_prior_zarr(path: Path) -> dict[str, np.ndarray]:
    root = zarr.open_group(path, mode="r")
    tree = _read_zarr_tree(root)
    return dict(tree["prior_samples"])


def write_stimuli_zarr(output: Path, stimuli: StimulusBatch) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(output, mode="w")
    _write_zarr_tree(root, _stimuli_to_tree(stimuli))


def read_stimuli_zarr(path: Path) -> StimulusBatch:
    root = zarr.open_group(path, mode="r")
    return _stimuli_from_tree(_read_zarr_tree(root))


def write_simulation_zarr(output: Path, simulation: SimulationBatch) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(output, mode="w")
    _write_zarr_tree(root, _simulation_to_tree(simulation), float_dtype=np.float16)


def read_simulation_zarr(path: Path) -> SimulationBatch:
    root = zarr.open_group(path, mode="r")
    return _simulation_from_tree(_read_zarr_tree(root))


# Posterior artifacts


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    value = jax.device_get(value)
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
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(output, mode="w")
    root.attrs["diagnostics"] = _json_ready(diagnostics)
    samples_group = root.create_group("posterior_samples")
    samples_group.attrs["dims"] = ["chain", "draw"]
    samples_group.attrs["dtype"] = "float16"
    for parameter, values in samples.items():
        data = _array_for_zarr(values, float_dtype=np.float16)
        chunks = (1, data.shape[1]) if data.ndim == 2 else data.shape
        samples_group.create_array(parameter, data=data, chunks=chunks)


def read_posterior_zarr(path: Path) -> dict[str, np.ndarray]:
    root = zarr.open_group(path, mode="r")
    tree = _read_zarr_tree(root)
    return dict(tree["posterior_samples"])


def read_posterior_diagnostics_zarr(path: Path) -> dict[str, Any]:
    root = zarr.open_group(path, mode="r")
    return dict(root.attrs.get("diagnostics", {}))
