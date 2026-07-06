from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from coupling_sources.core.data import MCMCSpecs
from coupling_sources.core.inference import run_inference
from coupling_sources.utils.artifacts import read_stimuli_zarr, write_posterior_zarr, write_simulation_zarr


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run assigned posterior jobs on one host.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-archive", type=Path, default=None)
    parser.add_argument("--assigned-jobs", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, required=True)
    parser.add_argument("--result-archive", type=Path, default=None)
    return parser.parse_args(argv)


def run_assigned_jobs(
    *,
    run_dir: Path,
    assigned_jobs_path: Path,
    max_parallel: int,
    result_archive_path: Path | None,
) -> int:
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")

    assignment = json.loads(assigned_jobs_path.read_text())
    jobs = list(assignment["jobs"])
    worker_label = _safe_path_part(assignment.get("worker_label", "local"))
    mcmc = MCMCSpecs(**assignment["mcmc"])
    progress_bar = bool(assignment.get("progress_bar", False))

    print(
        f"worker start: {worker_label}: jobs={len(jobs)} max_parallel={max_parallel}",
        flush=True,
    )
    failed = 0
    status_dir = run_dir / "job_status"
    status_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(
        max_workers=max_parallel,
        mp_context=get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(
                _run_one_job,
                str(run_dir),
                job_id,
                mcmc,
                progress_bar,
                worker_label,
            ): job_id
            for job_id in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            job_id = futures[future]
            try:
                status = future.result()
            except Exception:
                status = {
                    "job_id": job_id,
                    "ok": False,
                    "error": traceback.format_exc(),
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }
            _write_status(status_dir, status)
            if not status.get("ok", False):
                failed += 1
            print(
                f"worker progress: {worker_label}: {completed}/{len(jobs)} "
                f"job={job_id} ok={status.get('ok', False)}",
                flush=True,
            )

    if result_archive_path is not None:
        _write_result_archive(run_dir, jobs, worker_label, result_archive_path)
    print(f"worker done: {worker_label}: failed={failed}", flush=True)
    return 1 if failed else 0


def _run_one_job(
    run_dir: str,
    job_id: str,
    mcmc: MCMCSpecs,
    progress_bar: bool,
    worker_label: str,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    job_dir = run_path / "jobs" / job_id
    log_path = run_path / "worker_logs" / worker_label / f"{_safe_path_part(job_id)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", buffering=1) as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            print(f"fit start: {job_id}", flush=True)
            try:
                simulation_config = json.loads((job_dir / "simulation_parameters.json").read_text())
                result = run_inference(
                    stimuli=read_stimuli_zarr(job_dir / "stimuli.zarr"),
                    simulation_parameters=simulation_config["parameters"],
                    simulation_seed=int(simulation_config["seed"]),
                    mcmc=mcmc,
                    progress_bar=progress_bar,
                    progress_label=job_id,
                )
                write_simulation_zarr(job_dir / "simulation.zarr", result["simulation"])
                write_posterior_zarr(
                    job_dir / "posterior.zarr",
                    diagnostics=result["diagnostics"],
                    samples=result["samples"],
                )
                print(f"fit done: {job_id}", flush=True)
                return {
                    "job_id": job_id,
                    "ok": True,
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "posterior_path": str(Path("jobs") / job_id / "posterior.zarr"),
                    "log_path": str(log_path.relative_to(run_path)),
                    "chain_method": result["chain_method"],
                }
            except Exception:
                error = traceback.format_exc()
                print(error, file=sys.stderr, flush=True)
                return {
                    "job_id": job_id,
                    "ok": False,
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "log_path": str(log_path.relative_to(run_path)),
                    "error": error,
                }


def _write_status(status_dir: Path, status: dict[str, Any]) -> None:
    job_id = _safe_path_part(status["job_id"])
    (status_dir / f"{job_id}.json").write_text(json.dumps(status, indent=2) + "\n")


def _write_result_archive(
    run_dir: Path,
    job_ids: Sequence[str],
    worker_label: str,
    result_archive_path: Path,
) -> None:
    result_archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(result_archive_path, "w:gz") as archive:
        for job_id in job_ids:
            posterior_path = run_dir / "jobs" / job_id / "posterior.zarr"
            status_path = run_dir / "job_status" / f"{_safe_path_part(job_id)}.json"
            log_path = run_dir / "worker_logs" / worker_label / f"{_safe_path_part(job_id)}.log"
            _add_if_exists(archive, posterior_path, run_dir)
            _add_if_exists(archive, status_path, run_dir)
            _add_if_exists(archive, log_path, run_dir)


def _add_if_exists(archive: tarfile.TarFile, path: Path, base: Path) -> None:
    if path.exists():
        archive.add(path, arcname=path.relative_to(base))


def _safe_extract(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (output_root / member.name).resolve()
            if output_root != target and output_root not in target.parents:
                raise ValueError(f"unsafe archive member path: {member.name}")
        archive.extractall(output_root, members=members, filter="data")


def _safe_path_part(value: object) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return name or "job"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_archive is not None:
        _safe_extract(args.input_archive, args.run_dir)
    return run_assigned_jobs(
        run_dir=args.run_dir,
        assigned_jobs_path=args.assigned_jobs,
        max_parallel=args.max_parallel,
        result_archive_path=args.result_archive,
    )


if __name__ == "__main__":
    raise SystemExit(main())
