from __future__ import annotations

import signal
import tomllib
from pathlib import Path

import pytest

from coupling_sources import cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_registers_cluster_console_commands() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    assert project["project"]["scripts"] == {
        "coupling-sources-scheduler": "distributed.cli.dask_scheduler:main",
        "coupling-sources-worker": "distributed.cli.dask_worker:main",
        "coupling-sources-coordinator": "coupling_sources.cli:coordinator_main",
    }


def test_coordinator_command_delegates_to_parameter_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_parameter_recovery(**kwargs: object) -> int:
        captured.update(kwargs)
        return 7

    monkeypatch.setattr(cli, "run_parameter_recovery", fake_run_parameter_recovery)

    result = cli.coordinator_main(
        [
            "--config",
            "run.json",
            "--compute-config",
            "compute.json",
            "--artifacts-root",
            "output",
        ]
    )

    assert result == 7
    assert captured == {
        "config_path": Path("run.json"),
        "compute_config_path": Path("compute.json"),
        "artifacts_root": Path("output"),
    }


def test_coordinator_command_restores_sigterm_after_interruption(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt(**kwargs: object) -> int:
        del kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_parameter_recovery", interrupt)

    result = cli.coordinator_main(
        ["--config", "run.json", "--compute-config", "compute.json"]
    )

    assert result == 130
    assert signal.getsignal(signal.SIGTERM) == previous_sigterm
    assert "cancelling outstanding jobs" in capsys.readouterr().err
