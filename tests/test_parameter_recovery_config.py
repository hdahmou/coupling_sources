import json
from dataclasses import fields
from pathlib import Path

import pytest

from coupling_sources.compute.backend import ComputeConfig, _machine_env
from coupling_sources.core.data import MCMCSpecs
from coupling_sources.core.gain_calibration import GainCalibration
from coupling_sources.workflows import job_grid
from coupling_sources.workflows.parameter_recovery import _parallel_chains_per_fit


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "expected_jobs"),
    [
        ("configs/simple_fit/default.json", 1),
        ("configs/parameters_sweep/adam.json", 192),
    ],
)
def test_checked_run_configs_expand(
    relative_path: str,
    expected_jobs: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads((PROJECT_ROOT / relative_path).read_text())
    estimates = {item.name: 1.0 for item in fields(GainCalibration)}
    monkeypatch.setattr(
        job_grid,
        "calibrate_gain_constants",
        lambda spec: GainCalibration(**estimates),
    )

    assert len(job_grid.build_run_grid(config)) == expected_jobs


def test_parallel_chains_per_fit_reads_new_name() -> None:
    assert _parallel_chains_per_fit({"parallel_chains_per_fit": 4}) == 4


def test_mcmc_specs_normalize_config_values() -> None:
    specs = MCMCSpecs.from_mapping(
        {
            "seed": "1",
            "num_warmup": "2",
            "num_samples": "3",
            "num_chains": "4",
            "chain_method": "parallel",
            "target_accept_prob": "0.8",
            "max_tree_depth": "5",
        }
    )

    assert specs == MCMCSpecs(1, 2, 3, 4, "parallel", 0.8, 5)


def test_parallel_chains_per_fit_rejects_removed_name() -> None:
    with pytest.raises(ValueError, match="host_device_count.*removed"):
        _parallel_chains_per_fit({"host_device_count": 3})


def test_parallel_chains_per_fit_rejects_removed_name_even_with_new_name() -> None:
    with pytest.raises(ValueError, match="host_device_count.*removed"):
        _parallel_chains_per_fit(
            {
                "parallel_chains_per_fit": 4,
                "host_device_count": 3,
            }
        )


def test_machine_env_sets_parallel_chain_device_count() -> None:
    config = ComputeConfig.from_mapping(
        {
            "env": {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false"},
            "machines": [{"host": "local", "max_concurrent_fits": 1}],
        }
    )

    env = _machine_env(config.machines[0], config, parallel_chains_per_fit=4)

    assert (
        env["XLA_FLAGS"]
        == "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=4"
    )


def test_machine_env_enables_persistent_compilation_cache() -> None:
    config = ComputeConfig.from_mapping(
        {"machines": [{"host": "local", "max_concurrent_fits": 1}]}
    )

    env = _machine_env(config.machines[0], config)

    assert env["JAX_ENABLE_COMPILATION_CACHE"] == "true"
    assert env["JAX_COMPILATION_CACHE_DIR"] == ".jax_cache"


def test_machine_env_preserves_compilation_cache_override() -> None:
    config = ComputeConfig.from_mapping(
        {
            "env": {"JAX_COMPILATION_CACHE_DIR": "/tmp/shared-jax-cache"},
            "machines": [{"host": "local", "max_concurrent_fits": 1}],
        }
    )

    env = _machine_env(config.machines[0], config)

    assert env["JAX_COMPILATION_CACHE_DIR"] == "/tmp/shared-jax-cache"


def test_machine_env_replaces_machine_specific_device_count() -> None:
    config = ComputeConfig.from_mapping(
        {
            "env": {"XLA_FLAGS": "--xla_force_host_platform_device_count=2"},
            "machines": [
                {
                    "host": "local",
                    "max_concurrent_fits": 1,
                    "env": {"XLA_FLAGS": "--xla_force_host_platform_device_count=1"},
                }
            ],
        }
    )

    env = _machine_env(config.machines[0], config, parallel_chains_per_fit=4)

    assert env["XLA_FLAGS"] == "--xla_force_host_platform_device_count=4"
