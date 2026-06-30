from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import traceback
from collections.abc import Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import fields
from datetime import datetime
from pathlib import Path

from distributed import Client, LocalCluster, as_completed

from coupling_sources.parameters.mcmc import MCMCParams
from coupling_sources.parameters.simulation import SimParams
from coupling_sources.parameters.stimuli import StimuliParams
from artifacts import (
    read_simulation_zarr,
    write_posterior_zarr,
    write_prior_zarr,
    write_simulation_zarr,
    write_stimuli_zarr,
)
from generate_posterior import fit_posterior
from generate_prior import generate_prior
from generate_simulations import generate_simulation
from generate_stimuli import generate_stimulus
from job_grid import GridRun, build_run_grid


@contextmanager
def append_run_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log_file:
        stdout_fd = os.dup(sys.stdout.fileno())
        stderr_fd = os.dup(sys.stderr.fileno())
        try:
            os.dup2(log_file.fileno(), sys.stdout.fileno())
            os.dup2(log_file.fileno(), sys.stderr.fileno())
            with redirect_stdout(log_file), redirect_stderr(log_file):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            log_file.flush()
            os.dup2(stdout_fd, sys.stdout.fileno())
            os.dup2(stderr_fd, sys.stderr.fileno())
            os.close(stdout_fd)
            os.close(stderr_fd)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create posterior artifact trees from a config file.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compute-config", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=None)
    return parser.parse_args(argv)


def run_posterior_fit(
    *,
    simulation_path: Path,
    output: Path,
    params: MCMCParams,
    host_device_count: int | None,
    progress_bar: bool,
    log_path: Path,
    job_id: str,
) -> str:
    with append_run_log(log_path):
        print(f"fit start: {job_id} {simulation_path}", file=sys.stderr)
        posterior = fit_posterior(
            simulation=read_simulation_zarr(simulation_path),
            params=params,
            host_device_count=host_device_count,
            progress_bar=progress_bar,
            progress_label=job_id,
        )
        write_posterior_zarr(
            output,
            diagnostics=posterior["diagnostics"],
            samples=posterior["samples"],
            summary=posterior["summary"],
        )
    return f"{job_id}: {output}"


def local_compute_machine(compute_config: dict[str, object]) -> dict[str, object]:
    for machine in compute_config["machines"]:
        if machine["machine_label"] == "local":
            return machine
    raise ValueError("compute config must include a local machine")


def dispatch_posterior_fits(
    fits: list[dict[str, object]],
    compute_machine: dict[str, object],
    log_path: Path,
) -> None:
    if not fits:
        return

    if compute_machine["machine_label"] != "local":
        raise ValueError("only local compute is supported for now")

    with LocalCluster(
        n_workers=compute_machine["max_concurrent_fits"],
        threads_per_worker=1,
        env={"COUPLING_SOURCES_RUN_LOG": str(log_path)},
        preload=[str(Path(__file__).with_name("dask_log_preload.py"))],
    ) as cluster, Client(cluster) as client:
        futures = [client.submit(run_posterior_fit, **fit) for fit in fits]
        for future in as_completed(futures):
            print(f"fit complete: {future.result()}", file=sys.stderr)


def job_name_from_config(config_path: Path, started_at: datetime) -> str:
    resolved = config_path.resolve()
    try:
        name_path = resolved.relative_to(Path.cwd() / "configs")
    except ValueError:
        name_path = Path(resolved.stem)
    else:
        name_path = name_path.with_suffix("")

    config_name = "_".join(name_path.parts)
    config_name = re.sub(r"[^A-Za-z0-9]+", "_", config_name).strip("_").lower()
    return f"{config_name}_{started_at:%Y_%m_%d}"


def write_job_grid(output: Path, runs: Sequence[GridRun]) -> None:
    fieldnames = ["job_id", "stimuli_path", "simulation_path", "posterior_path"]
    fieldnames.extend(f"stimuli_{field.name}" for field in fields(StimuliParams))
    fieldnames.extend(f"simulation_{field.name}" for field in fields(SimParams))

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            job_path = Path("jobs") / run.run_id
            row = {
                "job_id": run.run_id,
                "stimuli_path": str(job_path / "stimuli.zarr"),
                "simulation_path": str(job_path / "simulation.zarr"),
                "posterior_path": str(job_path / "posterior.zarr"),
            }
            for key, value in run.stimuli.as_dict().items():
                if isinstance(value, list | tuple | dict):
                    value = json.dumps(value)
                row[f"stimuli_{key}"] = value
            for key, value in run.simulation.as_dict().items():
                row[f"simulation_{key}"] = value
            writer.writerow(row)


def prepare_job(
    *,
    job_dir: Path,
    stimuli_params: StimuliParams,
    simulation_seed: int,
    simulation_params: SimParams,
    mcmc_params: MCMCParams,
    host_device_count: int | None,
    progress_bar: bool,
    log_path: Path,
) -> dict[str, object]:
    job_dir.mkdir(parents=True, exist_ok=True)

    stimuli_path = job_dir / "stimuli.zarr"
    simulation_path = job_dir / "simulation.zarr"
    posterior_path = job_dir / "posterior.zarr"

    stimuli = generate_stimulus(stimuli_params)
    write_stimuli_zarr(stimuli_path, stimuli)
    simulation = generate_simulation(
        seed=simulation_seed,
        params=simulation_params,
        stimuli=stimuli,
    )
    write_simulation_zarr(simulation_path, simulation)
    print(f"prepared {job_dir}", file=sys.stderr)

    return {
        "simulation_path": simulation_path,
        "output": posterior_path,
        "params": mcmc_params,
        "host_device_count": host_device_count,
        "progress_bar": progress_bar,
        "log_path": log_path,
        "job_id": job_dir.name,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text())
    compute_config = json.loads(args.compute_config.read_text())

    started_at = datetime.now()
    job_config = config["job"]
    artifacts_root = args.artifacts_root or Path(job_config["artifacts_root"])
    diagnostic_prior_seed = job_config["diagnostic_prior_seed"]
    diagnostic_prior_draws = job_config["diagnostic_prior_draws"]
    base_job_name = job_name_from_config(args.config, started_at)
    base_job_dir = artifacts_root / base_job_name
    base_job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, base_job_dir / "config.json")
    log_path = (base_job_dir / "logs.log").resolve()

    with append_run_log(log_path):
        try:
            compute_machine = local_compute_machine(compute_config)
            runs = build_run_grid(config)
            simulation_config = config["simulation"]
            simulation_seed = simulation_config["seed"]
            mcmc_config = config["mcmc"]
            mcmc_params = MCMCParams.from_mapping(mcmc_config)
            host_device_count = mcmc_config["host_device_count"]
            progress_bar = mcmc_config["progress_bar"]

            prior_samples = generate_prior(diagnostic_prior_seed, diagnostic_prior_draws)
            write_prior_zarr(base_job_dir / "prior.zarr", prior_samples)
            write_job_grid(base_job_dir / "job_grid.csv", runs)

            posterior_fits: list[dict[str, object]] = []
            for run in runs:
                posterior_fits.append(
                    prepare_job(
                        job_dir=base_job_dir / "jobs" / run.run_id,
                        stimuli_params=run.stimuli,
                        simulation_seed=simulation_seed,
                        simulation_params=run.simulation,
                        mcmc_params=mcmc_params,
                        host_device_count=host_device_count,
                        progress_bar=progress_bar,
                        log_path=log_path,
                    )
                )

            dispatch_posterior_fits(posterior_fits, compute_machine, log_path)
            print(f"all jobs complete: {base_job_dir}", file=sys.stderr)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
