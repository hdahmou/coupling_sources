from __future__ import annotations

import argparse
import json
import signal
import sys
import tarfile
import traceback
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from threading import Event, current_thread, main_thread
from typing import Any

import numpy as np
import numpyro

from coupling_sources.compute.archives import safe_extract_tar
from coupling_sources.core.data import GainConstants, MCMCSpecs, StimulusSpec
from coupling_sources.core.inference import fit_simulation, simulate_for_inference
from coupling_sources.core.stimuli import generate_stimulus
from coupling_sources.utils.artifacts import Artifacts

POLL_SECONDS = 0.5


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
    shutdown_event: Event | None = None,
) -> int:
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")

    assignment = json.loads(assigned_jobs_path.read_text())
    jobs = list(assignment["jobs"])
    worker_label = str(assignment.get("worker_label", "local"))
    mcmc = MCMCSpecs.from_mapping(assignment["mcmc"])
    progress_bar = bool(assignment.get("progress_bar", False))

    print(
        f"worker start: {worker_label}: jobs={len(jobs)} max_parallel={max_parallel}",
        flush=True,
    )
    failed = 0

    executor = ProcessPoolExecutor(
        max_workers=max_parallel,
        mp_context=get_context("spawn"),
        max_tasks_per_child=1,
    )
    futures = {}
    try:
        futures = {
            executor.submit(
                _run_one_job,
                str(run_dir),
                job_id,
                mcmc,
                progress_bar,
            ): job_id
            for job_id in jobs
        }
        pending = set(futures)
        completed = 0
        while pending:
            if shutdown_event is not None and shutdown_event.is_set():
                raise KeyboardInterrupt
            done, pending = wait(
                pending,
                timeout=POLL_SECONDS,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                completed += 1
                job_id = futures[future]
                try:
                    status = future.result()
                except Exception:
                    status = {
                        "ok": False,
                        "error": traceback.format_exc(),
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                    }
                status_path = run_dir / "jobs" / job_id / "status.json"
                status_path.write_text(json.dumps(status, indent=2) + "\n")
                if not status.get("ok", False):
                    failed += 1
                print(
                    f"worker progress: {worker_label}: {completed}/{len(jobs)} "
                    f"job={job_id} ok={status.get('ok', False)}",
                    flush=True,
                )
    except BaseException:
        for future in futures:
            future.cancel()
        print(
            f"worker interrupted: {worker_label}; terminating outstanding jobs",
            file=sys.stderr,
            flush=True,
        )
        executor.kill_workers()
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=True)

    if result_archive_path is not None:
        _write_result_archive(run_dir, jobs, result_archive_path)
    print(f"worker done: {worker_label}: failed={failed}", flush=True)
    return 1 if failed else 0


def _run_one_job(
    run_dir: str,
    job_id: str,
    mcmc: MCMCSpecs,
    progress_bar: bool,
) -> dict[str, Any]:
    job_dir = Path(run_dir) / "jobs" / job_id
    log_path = job_dir / "job.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", buffering=1) as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            print(f"fit start: {job_id}", flush=True)
            try:
                numpyro.enable_x64(True)
                job_config = json.loads((job_dir / "job.json").read_text())
                stimuli = generate_stimulus(
                    StimulusSpec.from_mapping(job_config["stimulus"])
                )
                Artifacts.write_config(job_dir / "stimuli.npz", stimuli.to_config())

                gains = GainConstants.from_mapping(job_config["gain_constants"])
                simulation_config = job_config["simulation"]
                simulation = simulate_for_inference(
                    stimuli=stimuli,
                    simulation_parameters=simulation_config["parameters"],
                    simulation_seed=int(simulation_config["seed"]),
                    gains=gains,
                )
                Artifacts.write_config(
                    job_dir / "simulation.npz",
                    simulation.to_config(),
                    float_dtype=np.float16,
                )

                fit_result = fit_simulation(
                    simulation=simulation,
                    c0=simulation_config["parameters"]["c0"],
                    gains=gains,
                    mcmc=mcmc,
                    progress_bar=progress_bar,
                    progress_label=job_id,
                )
                Artifacts.write_dist_samples(
                    job_dir / "posterior.npz",
                    fit_result["samples"],
                )
                Artifacts.write_mcmc_diagnostics(
                    job_dir / "posterior.npz",
                    fit_result["diagnostics"],
                )
                print(f"fit done: {job_id}", flush=True)
                return {
                    "ok": True,
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "chain_method": fit_result["chain_method"],
                }
            except Exception:
                error = traceback.format_exc()
                print(error, file=sys.stderr, flush=True)
                return {
                    "ok": False,
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "error": error,
                }


def _write_result_archive(
    run_dir: Path,
    job_ids: Sequence[str],
    result_archive_path: Path,
) -> None:
    result_archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(result_archive_path, "w:gz") as archive:
        for job_id in job_ids:
            job_dir = run_dir / "jobs" / job_id
            for artifact_name in (
                "stimuli.npz",
                "simulation.npz",
                "posterior.npz",
                "status.json",
                "job.log",
            ):
                _add_if_exists(archive, job_dir / artifact_name, run_dir)


def _add_if_exists(archive: tarfile.TarFile, path: Path, base: Path) -> None:
    if path.exists():
        archive.add(path, arcname=path.relative_to(base))


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


@contextmanager
def _sigterm_as_keyboard_interrupt():
    if current_thread() is not main_thread():
        yield
        return

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_archive is not None:
        safe_extract_tar(args.input_archive, args.run_dir)
    with _sigterm_as_keyboard_interrupt():
        try:
            return run_assigned_jobs(
                run_dir=args.run_dir,
                assigned_jobs_path=args.assigned_jobs,
                max_parallel=args.max_parallel,
                result_archive_path=args.result_archive,
            )
        except KeyboardInterrupt:
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
