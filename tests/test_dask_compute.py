from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from dask.distributed import LocalCluster

from coupling_sources.compute import backend, worker
from coupling_sources.compute.backend import PosteriorJob
from coupling_sources.compute.worker import JobResult
from coupling_sources.core.data import GainConstants, MCMCSpecs, StimulusSpec
from coupling_sources.utils.artifacts import Artifacts


class _FakeFuture:
    def __init__(self, key: str, compute: Any) -> None:
        self.key = key
        self._compute = compute
        self.release_calls = 0
        self.cancel_calls = 0

    def result(self) -> Any:
        return self._compute()

    def release(self) -> None:
        self.release_calls += 1

    def cancel(self) -> None:
        self.cancel_calls += 1


class _FakeClient:
    instances: list["_FakeClient"] = []

    def __init__(self, scheduler_address: str) -> None:
        self.scheduler_address = scheduler_address
        self.submit_calls: list[dict[str, Any]] = []
        self.run_calls: list[tuple[Any, tuple[Any, ...]]] = []
        self.futures: list[_FakeFuture] = []
        self.__class__.instances.append(self)

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb

    def submit(
        self,
        task_fn: Any,
        *args: Any,
        key: str,
        pure: bool,
        retries: int,
    ) -> _FakeFuture:
        self.submit_calls.append(
            {
                "task_fn": task_fn,
                "args": args,
                "key": key,
                "pure": pure,
                "retries": retries,
            }
        )
        future = _FakeFuture(key, lambda: task_fn(*args))
        self.futures.append(future)
        return future

    def run(self, task_fn: Any, *args: Any) -> dict[str, Any]:
        self.run_calls.append((task_fn, args))
        return {}


def _write_job(tmp_path: Path, index: int) -> PosteriorJob:
    job_id = f"job_{index:06d}"
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps({"index": index}))
    return PosteriorJob(job_id, job_dir)


def _complete_artifacts(log_contents: bytes = b"log") -> dict[str, bytes]:
    return {
        name: log_contents if name == "job.log" else name.encode()
        for name in worker.ARTIFACT_NAMES
    }


def test_run_posterior_jobs_submits_every_fit_before_collecting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeClient.instances.clear()
    jobs = [_write_job(tmp_path, index) for index in range(3)]

    def fake_run_job(
        job_id: str,
        job_config: dict[str, Any],
        mcmc: MCMCSpecs,
        progress_bar: bool,
        env: dict[str, str],
        _run_token: str,
    ) -> JobResult:
        assert job_config["index"] == int(job_id.removeprefix("job_"))
        assert mcmc.num_samples == 1
        assert progress_bar is False
        assert env["JAX_NUM_CPU_DEVICES"] == "2"
        return JobResult(
            status={"ok": True, "chain_method": "sequential"},
            artifacts=_complete_artifacts(job_id.encode()),
        )

    def fake_as_completed(futures: Any) -> list[_FakeFuture]:
        client = _FakeClient.instances[-1]
        assert len(client.submit_calls) == len(jobs)
        return list(futures)

    monkeypatch.setattr(backend, "Client", _FakeClient)
    monkeypatch.setattr(backend, "as_completed", fake_as_completed)
    monkeypatch.setattr(backend, "run_job", fake_run_job)

    backend.run_posterior_jobs(
        compute_config={"scheduler_address": "inproc://tests"},
        jobs=jobs,
        mcmc=MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2),
        parallel_chains_per_fit=2,
        progress_bar=False,
    )

    client = _FakeClient.instances[-1]
    assert client.scheduler_address == "inproc://tests"
    assert len(client.submit_calls) == 3
    assert len({call["key"] for call in client.submit_calls}) == 3
    assert all(call["pure"] is False for call in client.submit_calls)
    assert all(call["retries"] == 0 for call in client.submit_calls)
    assert all(future.release_calls == 1 for future in client.futures)
    for job in jobs:
        assert json.loads((job.job_dir / "status.json").read_text())["ok"] is True
        assert (job.job_dir / "job.log").read_bytes() == job.job_id.encode()


def test_run_posterior_jobs_records_failures_and_finishes_other_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeClient.instances.clear()
    jobs = [_write_job(tmp_path, index) for index in range(3)]

    def fake_run_job(job_id: str, *_args: Any) -> JobResult:
        ok = job_id != "job_000001"
        return JobResult(
            status={"ok": ok, **({} if ok else {"error": "fit failed"})},
            artifacts=_complete_artifacts() if ok else {"job.log": b"log"},
        )

    monkeypatch.setattr(backend, "Client", _FakeClient)
    monkeypatch.setattr(backend, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(backend, "run_job", fake_run_job)

    with pytest.raises(RuntimeError, match="1 job.*job_000001"):
        backend.run_posterior_jobs(
            compute_config={"scheduler_address": "inproc://tests"},
            jobs=jobs,
            mcmc=MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2),
            progress_bar=False,
        )

    assert all((job.job_dir / "status.json").exists() for job in jobs)
    assert all(
        future.release_calls == 1 for future in _FakeClient.instances[-1].futures
    )


def test_run_posterior_jobs_converts_dask_exception_to_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeClient.instances.clear()
    job = _write_job(tmp_path, 0)

    def fail_job(*_args: Any) -> JobResult:
        raise OSError("worker disappeared")

    monkeypatch.setattr(backend, "Client", _FakeClient)
    monkeypatch.setattr(backend, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(backend, "run_job", fail_job)

    with pytest.raises(RuntimeError, match="1 job"):
        backend.run_posterior_jobs(
            compute_config={"scheduler_address": "inproc://tests"},
            jobs=[job],
            mcmc=MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2),
            progress_bar=False,
        )

    status = json.loads((job.job_dir / "status.json").read_text())
    assert status["ok"] is False
    assert "worker disappeared" in status["error"]


def test_run_posterior_jobs_rejects_incomplete_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeClient.instances.clear()
    job = _write_job(tmp_path, 0)
    monkeypatch.setattr(backend, "Client", _FakeClient)
    monkeypatch.setattr(backend, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        backend,
        "run_job",
        lambda *_args: JobResult(
            status={"ok": True},
            artifacts={"job.log": b"incomplete"},
        ),
    )

    with pytest.raises(RuntimeError, match="1 job"):
        backend.run_posterior_jobs(
            compute_config={"scheduler_address": "inproc://tests"},
            jobs=[job],
            mcmc=MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2),
            progress_bar=False,
        )

    status = json.loads((job.job_dir / "status.json").read_text())
    assert status["ok"] is False
    assert "without required artifacts" in status["error"]
    assert "posterior.npz" in status["error"]


def test_run_posterior_jobs_cancels_submitted_futures_when_submission_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingSubmitClient(_FakeClient):
        def submit(self, *args: Any, **kwargs: Any) -> _FakeFuture:
            if self.futures:
                raise ConnectionError("scheduler disconnected")
            return super().submit(*args, **kwargs)

    FailingSubmitClient.instances.clear()
    jobs = [_write_job(tmp_path, index) for index in range(2)]
    monkeypatch.setattr(backend, "Client", FailingSubmitClient)

    with pytest.raises(ConnectionError, match="scheduler disconnected"):
        backend.run_posterior_jobs(
            compute_config={"scheduler_address": "inproc://tests"},
            jobs=jobs,
            mcmc=MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2),
            progress_bar=False,
        )

    assert FailingSubmitClient.instances[-1].futures[0].cancel_calls == 1
    assert FailingSubmitClient.instances[-1].run_calls[0][0] is worker.cancel_run


def test_run_job_starts_clean_python_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            command = captured["command"]
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True)
            (output_dir / "status.json").write_text(json.dumps({"ok": True}))
            (output_dir / "job.log").write_bytes(b"child log")
            return b"", b""

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        captured["command"] = list(command)
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(
        worker.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    result = asyncio.run(
        worker.run_job(
            "job_000000",
            {"value": 1},
            MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2),
            False,
            {"JAX_NUM_CPU_DEVICES": "2"},
            "test-run",
        )
    )

    assert captured["command"][:3] == [
        sys.executable,
        "-m",
        "coupling_sources.compute.worker",
    ]
    assert captured["env"]["JAX_NUM_CPU_DEVICES"] == "2"
    assert captured["stdout"] == asyncio.subprocess.PIPE
    assert captured["stderr"] == asyncio.subprocess.PIPE
    assert result.status == {"ok": True}
    assert result.artifacts == {"job.log": b"child log"}


def test_run_job_terminates_child_when_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(terminated=False, killed=False)

    async def scenario() -> None:
        started = asyncio.Event()

        class BlockingProcess:
            returncode: int | None = None

            async def communicate(self) -> tuple[bytes, bytes]:
                started.set()
                await asyncio.Future()
                raise AssertionError("unreachable")

            def terminate(self) -> None:
                state.terminated = True
                self.returncode = -15

            def kill(self) -> None:
                state.killed = True
                self.returncode = -9

            async def wait(self) -> int:
                assert self.returncode is not None
                return self.returncode

        async def fake_create_subprocess_exec(
            *_command: str,
            **_kwargs: Any,
        ) -> BlockingProcess:
            return BlockingProcess()

        monkeypatch.setattr(
            worker.asyncio,
            "create_subprocess_exec",
            fake_create_subprocess_exec,
        )
        task = asyncio.create_task(
            worker.run_job(
                "job_000000",
                {"value": 1},
                MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2),
                False,
                {},
                "test-run",
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert state.terminated is True
    assert state.killed is False


def test_cancel_run_terminates_registered_children() -> None:
    state = SimpleNamespace(terminated=False)

    class RunningProcess:
        returncode: int | None = None

        def terminate(self) -> None:
            state.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            raise AssertionError("terminate should be sufficient")

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    process: Any = RunningProcess()
    other_process: Any = RunningProcess()
    worker._CHILD_PROCESSES[("target-run", "job_000000")] = process
    worker._CHILD_PROCESSES[("other-run", "job_000001")] = other_process
    try:
        terminated = asyncio.run(worker.cancel_run("target-run"))
    finally:
        worker._CHILD_PROCESSES.clear()

    assert terminated == 1
    assert state.terminated is True


def test_run_posterior_jobs_round_trips_through_process_worker(
    tmp_path: Path,
) -> None:
    job = _write_job(tmp_path, 0)
    with LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
    ) as cluster:
        with pytest.raises(RuntimeError, match="1 job"):
            backend.run_posterior_jobs(
                compute_config={"scheduler_address": cluster.scheduler_address},
                jobs=[job],
                mcmc=MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2),
                progress_bar=False,
            )

    status = json.loads((job.job_dir / "status.json").read_text())
    assert status["ok"] is False
    assert "stimulus" in status["error"]
    assert (job.job_dir / "job.log").exists()


class _FakeStimuli:
    def to_config(self) -> dict[str, list[int]]:
        return {"stimuli": [1]}


class _FakeSimulation:
    def to_config(self) -> dict[str, list[int]]:
        return {"simulation": [1]}


def test_run_one_job_generates_simulates_saves_then_fits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    stimuli = _FakeStimuli()
    simulation = _FakeSimulation()
    monkeypatch.setattr(worker.numpyro, "enable_x64", lambda value: None)

    def fake_generate(spec: StimulusSpec) -> _FakeStimuli:
        assert spec.coupled_sources == (0, 1)
        events.append("generate")
        return stimuli

    def fake_simulate(**kwargs: Any) -> _FakeSimulation:
        assert kwargs["stimuli"] is stimuli
        assert kwargs["gains"] == GainConstants(19.946, 28.121)
        events.append("simulate")
        return simulation

    def fake_fit(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["simulation"] is simulation
        events.append("fit")
        return {
            "samples": {"sample": np.array([1.0])},
            "diagnostics": {"ok": True},
            "chain_method": "sequential",
        }

    monkeypatch.setattr(worker, "generate_stimulus", fake_generate)
    monkeypatch.setattr(worker, "simulate_for_inference", fake_simulate)
    monkeypatch.setattr(worker, "fit_simulation", fake_fit)

    status = worker._run_one_job(
        tmp_path,
        "job_000000",
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
            "gain_constants": {"g_star_y": 19.946, "g_star_z": 28.121},
            "simulation": {
                "seed": 10,
                "parameters": {"value": 1.0, "c0": 0.25},
            },
        },
        MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2),
        False,
    )

    assert status["ok"] is True
    assert events == ["generate", "simulate", "fit"]
    assert set(path.name for path in tmp_path.iterdir()) == {
        "stimuli.npz",
        "simulation.npz",
        "posterior.npz",
        "job.log",
    }
    assert Artifacts.read_dist_samples(tmp_path / "posterior.npz")["sample"] == 1.0
    assert Artifacts.read_mcmc_diagnostics(tmp_path / "posterior.npz") == {"ok": True}


def test_run_one_job_returns_partial_artifacts_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(worker.numpyro, "enable_x64", lambda value: None)
    monkeypatch.setattr(worker, "generate_stimulus", lambda spec: _FakeStimuli())
    monkeypatch.setattr(
        worker,
        "simulate_for_inference",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("simulation failed")),
    )

    status = worker._run_one_job(
        tmp_path,
        "job_000000",
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
            "gain_constants": {"g_star_y": 19.946, "g_star_z": 28.121},
            "simulation": {"seed": 10, "parameters": {"c0": 0.25}},
        },
        MCMCSpecs(1, 1, 1, 1, "sequential", 0.8, 2),
        False,
    )

    assert status["ok"] is False
    assert "simulation failed" in status["error"]
    assert (tmp_path / "stimuli.npz").exists()
    assert (tmp_path / "job.log").exists()
    assert not (tmp_path / "posterior.npz").exists()
