from __future__ import annotations

import json
import os
import re
import shutil
import sys
import traceback
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from coupling_sources.core.inference import fit_posterior
from coupling_sources.core.parameter_prior import sample_parameter_prior
from coupling_sources.core.parameters.mcmc import MCMCParams
from coupling_sources.core.parameters.simulation import SimParams
from coupling_sources.core.parameters.stimuli import StimuliParams
from coupling_sources.core.simulation import SimulationArtifact, generate_simulation
from coupling_sources.core.stimuli import generate_stimulus
from coupling_sources.compute.backend import add_prompted_ssh_password, run_tasks
from coupling_sources.utils.artifacts import (
    read_simulation_zarr,
    write_posterior_zarr,
    write_prior_zarr,
    write_simulation_zarr,
    write_stimuli_zarr,
)
from coupling_sources.workflows.job_grid import build_run_grid, write_job_grid


@contextmanager
def _append_run_log(log_path: Path):
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


def _xla_host_device_count(mcmc_config: dict[str, object]) -> int | None:
    configured = mcmc_config.get("host_device_count")
    if configured is not None:
        count = int(configured)
    elif mcmc_config.get("chain_method") == "parallel" and int(mcmc_config["num_chains"]) > 1:
        count = int(mcmc_config["num_chains"])
    else:
        return None

    if count < 1:
        raise ValueError("host_device_count must be a positive integer")
    return count


def _set_worker_xla_host_device_count(
    compute_config: dict[str, object], count: int | None
) -> None:
    if count is None:
        return

    env = compute_config.setdefault("env", {})
    if not isinstance(env, dict):
        raise ValueError("compute config env must be an object")

    flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+", "", str(env.get("XLA_FLAGS", ""))
    ).split()
    env["XLA_FLAGS"] = " ".join(
        [f"--xla_force_host_platform_device_count={count}", *flags]
    )


def _safe_log_name(value: object) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return name or "job"


@contextmanager
def _task_worker_log(job_id: str):
    log_dir = os.environ.get("COUPLING_SOURCES_WORKER_LOG_DIR")
    if log_dir is None:
        with nullcontext():
            yield
        return

    log_path = Path(log_dir) / f"{_safe_log_name(job_id)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log_file:
        stdout_fd = os.dup(sys.stdout.fileno())
        stderr_fd = os.dup(sys.stderr.fileno())
        try:
            os.dup2(log_file.fileno(), sys.stdout.fileno())
            os.dup2(log_file.fileno(), sys.stderr.fileno())
            with redirect_stdout(log_file), redirect_stderr(log_file):
                print(f"worker task log start: {job_id}", file=sys.stderr)
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            log_file.flush()
            os.dup2(stdout_fd, sys.stdout.fileno())
            os.dup2(stderr_fd, sys.stderr.fileno())
            os.close(stdout_fd)
            os.close(stderr_fd)


@dataclass(frozen=True)
class PosteriorFit:
    output: Path
    simulation: SimulationArtifact
    params: MCMCParams
    host_device_count: int | None
    progress_bar: bool
    job_id: str

    def task_kwargs(self) -> dict[str, object]:
        return {
            "simulation": self.simulation,
            "params": self.params,
            "host_device_count": self.host_device_count,
            "progress_bar": self.progress_bar,
            "job_id": self.job_id,
        }


def _run_inference_job(
    *,
    simulation: SimulationArtifact,
    params: MCMCParams,
    host_device_count: int | None,
    progress_bar: bool,
    job_id: str,
) -> dict[str, object]:
    with _task_worker_log(job_id):
        print(f"fit start: {job_id}", file=sys.stderr)
        posterior = fit_posterior(
            simulation=simulation,
            params=params,
            host_device_count=host_device_count,
            progress_bar=progress_bar,
            progress_label=job_id,
        )
        print(f"fit done: {job_id}", file=sys.stderr)
    return {"job_id": job_id, **posterior}


def _dispatch_posterior_fits(
    fits: list[PosteriorFit],
    compute_config: dict[str, object],
    log_path: Path,
) -> None:
    if not fits:
        return

    submissions = [(fit.output, fit.task_kwargs()) for fit in fits]
    for output, result in run_tasks(compute_config, log_path, _run_inference_job, submissions):
        posterior = result
        write_posterior_zarr(
            output,
            diagnostics=posterior["diagnostics"],
            samples=posterior["samples"],
        )
        print(f"fit complete: {posterior['job_id']}: {output}", file=sys.stderr)


def _job_name_from_config(config_path: Path, started_at: datetime) -> str:
    resolved = config_path.resolve()
    try:
        name_path = resolved.relative_to(Path.cwd() / "configs")
    except ValueError:
        name_path = Path(resolved.stem)
    else:
        name_path = name_path.with_suffix("")

    config_name = "_".join(name_path.parts)
    config_name = re.sub(r"[^A-Za-z0-9]+", "_", config_name).strip("_").lower()
    return f"{config_name}_{started_at:%Y_%m_%d_%H_%M_%S}"


def _prepare_job(
    *,
    job_dir: Path,
    stimuli_params: StimuliParams,
    simulation_seed: int,
    simulation_params: SimParams,
    mcmc_params: MCMCParams,
    host_device_count: int | None,
    progress_bar: bool,
) -> PosteriorFit:
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

    return PosteriorFit(
        output=posterior_path,
        simulation=read_simulation_zarr(simulation_path),
        params=mcmc_params,
        host_device_count=host_device_count,
        progress_bar=progress_bar,
        job_id=job_dir.name,
    )


def run_parameter_recovery(
    *,
    config_path: Path,
    compute_config_path: Path,
    artifacts_root: Path | None = None,
) -> int:
    config = json.loads(config_path.read_text())
    compute_config = json.loads(compute_config_path.read_text())
    add_prompted_ssh_password(compute_config)

    started_at = datetime.now()
    job_config = config["job"]
    artifacts_root = artifacts_root or Path(job_config["artifacts_root"])
    diagnostic_prior_seed = job_config["diagnostic_prior_seed"]
    diagnostic_prior_draws = job_config["diagnostic_prior_draws"]
    base_job_name = _job_name_from_config(config_path, started_at)
    base_job_dir = artifacts_root / base_job_name
    base_job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, base_job_dir / "config.json")
    log_path = (base_job_dir / "logs.log").resolve()

    with _append_run_log(log_path):
        try:
            runs = build_run_grid(config)
            simulation_config = config["simulation"]
            simulation_seed = simulation_config["seed"]
            mcmc_config = config["mcmc"]
            host_device_count = _xla_host_device_count(mcmc_config)
            _set_worker_xla_host_device_count(compute_config, host_device_count)
            mcmc_params = MCMCParams.from_mapping(mcmc_config)
            progress_bar = mcmc_config["progress_bar"]

            prior_samples = sample_parameter_prior(diagnostic_prior_seed, diagnostic_prior_draws)
            write_prior_zarr(base_job_dir / "prior.zarr", prior_samples)
            write_job_grid(base_job_dir / "job_grid.csv", runs)

            posterior_fits: list[PosteriorFit] = []
            for run in runs:
                posterior_fits.append(
                    _prepare_job(
                        job_dir=base_job_dir / "jobs" / run.run_id,
                        stimuli_params=run.stimuli,
                        simulation_seed=simulation_seed,
                        simulation_params=run.simulation,
                        mcmc_params=mcmc_params,
                        host_device_count=host_device_count,
                        progress_bar=progress_bar,
                    )
                )

            _dispatch_posterior_fits(posterior_fits, compute_config, log_path)
            print(f"all jobs complete: {base_job_dir}", file=sys.stderr)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return 1
    return 0
