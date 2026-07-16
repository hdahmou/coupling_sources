from __future__ import annotations

import json
import tarfile
from dataclasses import asdict
from threading import Event

from coupling_sources.compute import fabric_worker
from coupling_sources.compute.backend import (
    ComputeConfig,
    HostResult,
    PosteriorJob,
    _PosteriorJobRunner,
)
from coupling_sources.core.data import GainConstants, MCMCSpecs, StimulusSpec
from coupling_sources.utils.artifacts import Artifacts


class _FakeStimuli:
    def to_config(self) -> dict[str, list[int]]:
        return {"stimuli": [1]}


class _FakeSimulation:
    def to_config(self) -> dict[str, list[int]]:
        return {"simulation": [1]}


def test_run_assigned_jobs_uses_one_process_per_job(monkeypatch, tmp_path) -> None:
    assignment_path = tmp_path / "assigned_jobs.json"
    mcmc = MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2)
    assignment_path.write_text(
        json.dumps(
            {
                "jobs": [],
                "worker_label": "local",
                "mcmc": asdict(mcmc),
                "progress_bar": False,
            }
        )
    )
    captured = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def shutdown(self, **kwargs):
            return None

    monkeypatch.setattr(fabric_worker, "ProcessPoolExecutor", FakeExecutor)

    result = fabric_worker.run_assigned_jobs(
        run_dir=tmp_path,
        assigned_jobs_path=assignment_path,
        max_parallel=3,
        result_archive_path=None,
    )

    assert result == 0
    assert captured["max_workers"] == 3
    assert captured["max_tasks_per_child"] == 1
    assert captured["mp_context"].get_start_method() == "spawn"


def test_run_one_job_generates_simulates_saves_then_fits(monkeypatch, tmp_path) -> None:
    job_id = "job_000000"
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "stimulus": {
                    "seed": 1,
                    "n_participants": 2,
                    "n_sources": 4,
                    "n_rounds": 1,
                    "n_trials_full": 4,
                    "n_trials_partial": 4,
                    "sigma": 0.3,
                    "lam": 0.7,
                    "coupled_sources": [0, 1],
                    "resultant_floor_full": 0.6,
                    "resultant_floor_leave_one_out": 0.6,
                },
                "gain_constants": {
                    "g_star_y": 19.946,
                    "g_star_z": 28.121,
                },
                "simulation": {
                    "seed": 10,
                    "parameters": {"value": 1.0, "c0": 0.25},
                },
            }
        )
    )
    events = []
    stimuli = _FakeStimuli()
    simulation = _FakeSimulation()

    monkeypatch.setattr(fabric_worker.numpyro, "enable_x64", lambda value: None)

    def fake_generate(spec):
        assert isinstance(spec, StimulusSpec)
        assert spec.coupled_sources == (0, 1)
        events.append("generate")
        return stimuli

    def fake_simulate(**kwargs):
        assert kwargs == {
            "stimuli": stimuli,
            "simulation_parameters": {"value": 1.0, "c0": 0.25},
            "simulation_seed": 10,
            "gains": GainConstants(g_star_y=19.946, g_star_z=28.121),
        }
        events.append("simulate")
        return simulation

    def fake_write_config(path, config, *, float_dtype=None):
        events.append(f"write:{path.name}")

    def fake_fit(**kwargs):
        assert kwargs["simulation"] is simulation
        assert kwargs["c0"] == 0.25
        assert kwargs["gains"] == GainConstants(
            g_star_y=19.946,
            g_star_z=28.121,
        )
        assert events == [
            "generate",
            "write:stimuli.npz",
            "simulate",
            "write:simulation.npz",
        ]
        events.append("fit")
        return {
            "samples": {"sample": [1.0]},
            "diagnostics": {"ok": True},
            "chain_method": "sequential",
        }

    monkeypatch.setattr(fabric_worker, "generate_stimulus", fake_generate)
    monkeypatch.setattr(fabric_worker, "simulate_for_inference", fake_simulate)
    monkeypatch.setattr(fabric_worker, "fit_simulation", fake_fit)
    monkeypatch.setattr(Artifacts, "write_config", fake_write_config)
    monkeypatch.setattr(
        Artifacts,
        "write_dist_samples",
        lambda path, samples: events.append("write:posterior.npz"),
    )
    monkeypatch.setattr(
        Artifacts,
        "write_mcmc_diagnostics",
        lambda path, diagnostics: events.append("write:diagnostics"),
    )

    status = fabric_worker._run_one_job(
        str(tmp_path),
        job_id,
        MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2),
        False,
    )

    assert status["ok"] is True
    assert "job_id" not in status
    assert "posterior_path" not in status
    assert "log_path" not in status
    assert events == [
        "generate",
        "write:stimuli.npz",
        "simulate",
        "write:simulation.npz",
        "fit",
        "write:posterior.npz",
        "write:diagnostics",
    ]


def test_result_archive_includes_all_generated_artifacts(tmp_path) -> None:
    job_id = "job_000000"
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    for name in (
        "stimuli.npz",
        "simulation.npz",
        "posterior.npz",
        "status.json",
        "job.log",
    ):
        (job_dir / name).write_bytes(name.encode())
    archive_path = tmp_path / "results.tar.gz"

    fabric_worker._write_result_archive(
        tmp_path,
        [job_id],
        archive_path,
    )

    with tarfile.open(archive_path, "r:gz") as archive:
        assert set(archive.getnames()) == {
            f"jobs/{job_id}/stimuli.npz",
            f"jobs/{job_id}/simulation.npz",
            f"jobs/{job_id}/posterior.npz",
            f"jobs/{job_id}/status.json",
            f"jobs/{job_id}/job.log",
        }


def test_input_archive_contains_only_job_config(tmp_path) -> None:
    run_dir = tmp_path / "run"
    job_dir = run_dir / "jobs" / "job_000000"
    job_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}")
    (run_dir / "job_grid.csv").write_text("job_id\njob_000000\n")
    (job_dir / "job.json").write_text("{}")
    config = ComputeConfig.from_mapping(
        {"machines": [{"host": "local", "max_concurrent_fits": 1}]}
    )
    machine = config.machines[0]
    job = PosteriorJob("job_000000", job_dir)
    runner = _PosteriorJobRunner(
        config=config,
        run_dir=run_dir,
        mcmc=MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2),
        parallel_chains_per_fit=1,
        progress_bar=False,
        shutdown_event=Event(),
    )
    assignment_path = runner._write_assignment(machine, [job])

    archive_path = runner._make_input_archive(machine, [job], assignment_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        assert set(archive.getnames()) == {
            "config.json",
            "job_grid.csv",
            "transfer/assigned_jobs_local.json",
            "jobs/job_000000/job.json",
        }


def test_host_result_reads_job_local_status(tmp_path) -> None:
    jobs = []
    for index, ok in enumerate((True, False)):
        job_id = f"job_{index:06d}"
        job_dir = tmp_path / "jobs" / job_id
        job_dir.mkdir(parents=True)
        (job_dir / "status.json").write_text(json.dumps({"ok": ok}))
        jobs.append(PosteriorJob(job_id, job_dir))

    result = HostResult.from_jobs(
        host="local",
        jobs=jobs,
        exit_code=1,
    )

    assert result.failed_jobs == ("job_000001",)
