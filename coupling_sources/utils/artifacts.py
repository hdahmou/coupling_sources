"""Small NPZ artifact store.

Artifacts are single ``.npz`` files. Data arrays use slash-separated keys for
nested mappings; metadata lives as JSON in one reserved key.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np

Tree = dict[str, Any]


class Artifacts:
    ATTRS_KEY = "__attrs_json__"
    KEY_SEPARATOR = "/"
    MCMC_DIAGNOSTICS_KEY = "mcmc_diagnostics"

    @classmethod
    def write_dist_samples(
        cls,
        path: Path,
        samples: Mapping[str, Any],
        *,
        float_dtype: Any = np.float16,
    ) -> None:
        cls._write(path, samples, attrs={}, float_dtype=float_dtype)

    @classmethod
    def read_dist_samples(cls, path: Path) -> dict[str, np.ndarray]:
        samples, _ = cls._read(path)
        return samples

    @classmethod
    def write_config(
        cls,
        path: Path,
        config: Mapping[str, Any],
        *,
        float_dtype: Any = None,
    ) -> None:
        cls._write(path, config, attrs={}, float_dtype=float_dtype)

    @classmethod
    def read_config(cls, path: Path) -> Tree:
        config, _ = cls._read(path)
        return config

    @classmethod
    def write_mcmc_diagnostics(
        cls,
        path: Path,
        diagnostics: Mapping[str, Any],
    ) -> None:
        samples, attrs = cls._read(path)
        attrs[cls.MCMC_DIAGNOSTICS_KEY] = cls._json_ready(diagnostics)
        cls._write(path, samples, attrs=attrs)

    @classmethod
    def read_mcmc_diagnostics(cls, path: Path) -> dict[str, Any]:
        _, attrs = cls._read(path)
        return dict(attrs[cls.MCMC_DIAGNOSTICS_KEY])

    @classmethod
    def _write(
        cls,
        path: Path,
        tree: Mapping[str, Any],
        *,
        attrs: Mapping[str, Any],
        float_dtype: Any = None,
    ) -> None:
        path = Path(path)
        cls._validate_path(path)
        if path.exists() and path.is_dir():
            raise ValueError(
                f"artifact path is a directory, expected .npz file: {path}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = cls._flatten_tree(tree, float_dtype=float_dtype)
        arrays[cls.ATTRS_KEY] = np.array(json.dumps(cls._json_ready(attrs)))
        with path.open("wb") as file:
            np.savez_compressed(file, allow_pickle=False, **arrays)

    @classmethod
    def _read(cls, path: Path) -> tuple[Tree, dict[str, Any]]:
        path = Path(path)
        cls._validate_path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as archive:
            attrs_json = archive[cls.ATTRS_KEY].item()
            arrays = {
                key: np.asarray(archive[key])
                for key in archive.files
                if key != cls.ATTRS_KEY
            }
        return cls._unflatten_tree(arrays), json.loads(attrs_json)

    @staticmethod
    def _validate_path(path: Path) -> None:
        if path.suffix != ".npz":
            raise ValueError(f"artifact path must end with .npz: {path}")

    @classmethod
    def _array_for_storage(cls, value: Any, *, float_dtype: Any = None) -> np.ndarray:
        data = np.asarray(jax.device_get(value))
        if float_dtype is not None and np.issubdtype(data.dtype, np.floating):
            return data.astype(float_dtype, copy=False)
        return data

    @classmethod
    def _flatten_tree(
        cls,
        tree: Mapping[str, Any],
        *,
        float_dtype: Any = None,
        prefix: str = "",
    ) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for name, value in tree.items():
            if value is None:
                continue
            key = f"{prefix}{cls.KEY_SEPARATOR}{name}" if prefix else name
            if isinstance(value, Mapping):
                arrays.update(
                    cls._flatten_tree(value, float_dtype=float_dtype, prefix=key)
                )
            else:
                arrays[key] = cls._array_for_storage(value, float_dtype=float_dtype)
        return arrays

    @classmethod
    def _unflatten_tree(cls, arrays: Mapping[str, np.ndarray]) -> Tree:
        tree: Tree = {}
        for key, value in arrays.items():
            parts = key.split(cls.KEY_SEPARATOR)
            node = tree
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return tree

    @classmethod
    def _json_ready(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: cls._json_ready(item) for key, item in value.items()}
        value = jax.device_get(value)
        if isinstance(value, list | tuple):
            return [cls._json_ready(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
