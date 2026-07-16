from __future__ import annotations

import json
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dask.distributed import Client, as_completed

from coupling_sources.compute.worker import (
    ARTIFACT_NAMES,
    JobResult,
    cancel_run,
    run_job,
)
from coupling_sources.core.data import MCMCSpecs

DEFAULT_JAX_COMPILATION_CACHE_DIR = ".jax_cache"


def _config_mapping(value: object | None, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class ComputeConfig:
    scheduler_address: str
    env: dict[str, object] | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ComputeConfig":
        raw_scheduler_address = values.get("scheduler_address")
        if (
            not isinstance(raw_scheduler_address, str)
            or not raw_scheduler_address.strip()
        ):
            raise ValueError("compute config must include scheduler_address")
        scheduler_address = raw_scheduler_address.strip()
        return cls(
            scheduler_address=scheduler_address,
            env=_config_mapping(values.get("env"), "env"),
        )


@dataclass(frozen=True)
class PosteriorJob:
    job_id: str
    job_dir: Path


def run_posterior_jobs(
    *,
    compute_config: dict[str, object],
    jobs: Sequence[PosteriorJob],
    mcmc: MCMCSpecs,
    parallel_chains_per_fit: int | None = None,
    progress_bar: bool,
) -> None:
    if not jobs:
        return

    config = ComputeConfig.from_mapping(compute_config)
    worker_env = _worker_env(config, parallel_chains_per_fit)
    run_token = uuid4().hex
    failed: list[str] = []

    with Client(config.scheduler_address) as client:
        print(f"connected to Dask scheduler: {config.scheduler_address}", flush=True)
        futures = {}
        completed_count = 0
        try:
            for job in jobs:
                future = client.submit(
                    run_job,
                    job.job_id,
                    json.loads((job.job_dir / "job.json").read_text()),
                    mcmc,
                    progress_bar,
                    worker_env,
                    run_token,
                    key=f"posterior-{run_token}-{job.job_id}",
                    pure=False,
                    retries=0,
                )
                futures[future] = job
            print(f"submitted {len(futures)} posterior job(s)", flush=True)

            for future in as_completed(futures):
                job = futures[future]
                try:
                    try:
                        result = future.result()
                        if not isinstance(result, JobResult):
                            raise TypeError(
                                f"Dask task for {job.job_id} returned "
                                f"{type(result).__name__}, expected JobResult"
                            )
                    except Exception:
                        result = JobResult(
                            status={
                                "ok": False,
                                "finished_at": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                                "error": traceback.format_exc(),
                            },
                            artifacts={},
                        )

                    result = _validate_job_result(job, result)
                    _persist_job_result(job, result)
                    completed_count += 1
                    ok = bool(result.status.get("ok", False))
                    if not ok:
                        failed.append(job.job_id)
                    print(
                        f"Dask progress: {completed_count}/{len(jobs)} "
                        f"job={job.job_id} ok={ok}",
                        flush=True,
                    )
                finally:
                    future.release()
        except BaseException:
            try:
                client.run(cancel_run, run_token)
            except Exception:
                print("failed to terminate fit subprocesses", flush=True)
                traceback.print_exc()
            finally:
                for future in futures:
                    future.cancel()
            raise

    if failed:
        raise RuntimeError(
            f"posterior fits failed for {len(failed)} job(s): {failed[:10]}"
        )


def _persist_job_result(job: PosteriorJob, result: JobResult) -> None:
    unknown_artifacts = set(result.artifacts) - set(ARTIFACT_NAMES)
    if unknown_artifacts:
        raise ValueError(
            f"job {job.job_id} returned unknown artifacts: "
            f"{sorted(unknown_artifacts)}"
        )

    job.job_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in result.artifacts.items():
        if not isinstance(contents, bytes):
            raise TypeError(f"job artifact {name} must contain bytes")
        (job.job_dir / name).write_bytes(contents)
    (job.job_dir / "status.json").write_text(json.dumps(result.status, indent=2) + "\n")


def _validate_job_result(job: PosteriorJob, result: JobResult) -> JobResult:
    ok = result.status.get("ok")
    error: str | None = None
    if not isinstance(ok, bool):
        error = f"job {job.job_id} returned a non-boolean ok status"
    elif ok:
        missing_artifacts = set(ARTIFACT_NAMES) - set(result.artifacts)
        if missing_artifacts:
            error = (
                f"job {job.job_id} reported success without required artifacts: "
                f"{sorted(missing_artifacts)}"
            )

    if error is None:
        return result

    status = dict(result.status)
    status["ok"] = False
    status["error"] = error
    status.setdefault(
        "finished_at",
        datetime.now().isoformat(timespec="seconds"),
    )
    return JobResult(status=status, artifacts=result.artifacts)


def _worker_env(
    config: ComputeConfig,
    parallel_chains_per_fit: int | None = None,
) -> dict[str, str]:
    env = {str(key): str(value) for key, value in (config.env or {}).items()}
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "true")
    env.setdefault("JAX_COMPILATION_CACHE_DIR", DEFAULT_JAX_COMPILATION_CACHE_DIR)
    if parallel_chains_per_fit is not None:
        if parallel_chains_per_fit < 1:
            raise ValueError("parallel_chains_per_fit must be positive")
        env["JAX_NUM_CPU_DEVICES"] = str(parallel_chains_per_fit)
    return env
