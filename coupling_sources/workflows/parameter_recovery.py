from __future__ import annotations

import json
import os
import re
import shutil
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from coupling_sources.compute.backend import (
    PosteriorJob,
    add_prompted_ssh_password,
    run_posterior_jobs,
)
from coupling_sources.core.data import MCMCSpecs
from coupling_sources.core.priors import CouplingPriors
from coupling_sources.utils.artifacts import Artifacts
from coupling_sources.workflows.job_grid import GridRun, build_run_grid, write_job_grid

PARALLEL_CHAINS_PER_FIT_KEY = "parallel_chains_per_fit"
REMOVED_HOST_DEVICE_COUNT_KEY = "host_device_count"


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
    run_dir: Path,
    run: GridRun,
    simulation_seed: int,
) -> PosteriorJob:
    job_dir = run_dir / "jobs" / run.run_id
    job_dir.mkdir(parents=True, exist_ok=True)

    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "stimulus": asdict(run.stimuli),
                "simulation": {
                    "seed": simulation_seed,
                    "parameters": run.simulation,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"prepared {job_dir}", file=sys.stderr)

    return PosteriorJob(job_id=run.run_id, job_dir=job_dir)


def _parallel_chains_per_fit(mcmc_config: dict[str, object]) -> int | None:
    if REMOVED_HOST_DEVICE_COUNT_KEY in mcmc_config:
        raise ValueError(
            f"mcmc.{REMOVED_HOST_DEVICE_COUNT_KEY} was removed; "
            f"use mcmc.{PARALLEL_CHAINS_PER_FIT_KEY}"
        )
    value = mcmc_config.get(PARALLEL_CHAINS_PER_FIT_KEY)
    if value is None:
        return None
    if not isinstance(value, int | float | str):
        raise ValueError(f"mcmc.{PARALLEL_CHAINS_PER_FIT_KEY} must be an integer")

    count = int(value)
    if count < 1:
        raise ValueError(f"mcmc.{PARALLEL_CHAINS_PER_FIT_KEY} must be positive")
    return count


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
            mcmc = MCMCSpecs(
                seed=int(mcmc_config["seed"]),
                num_warmup=int(mcmc_config["num_warmup"]),
                num_samples=int(mcmc_config["num_samples"]),
                num_chains=int(mcmc_config["num_chains"]),
                chain_method=str(mcmc_config["chain_method"]),
                target_accept_prob=float(mcmc_config["target_accept_prob"]),
                max_tree_depth=int(mcmc_config["max_tree_depth"]),
            )
            parallel_chains_per_fit = _parallel_chains_per_fit(mcmc_config)
            progress_bar = bool(mcmc_config["progress_bar"])

            prior_samples = CouplingPriors.sample_parameter_draws(
                diagnostic_prior_seed,
                diagnostic_prior_draws,
                simulation_config["c0"],
            )
            Artifacts.write_dist_samples(base_job_dir / "prior.npz", prior_samples)
            write_job_grid(base_job_dir / "job_grid.csv", runs)

            posterior_jobs = [
                _prepare_job(
                    run_dir=base_job_dir,
                    run=run,
                    simulation_seed=simulation_seed,
                )
                for run in runs
            ]

            run_posterior_jobs(
                compute_config=compute_config,
                run_dir=base_job_dir,
                jobs=posterior_jobs,
                mcmc=mcmc,
                parallel_chains_per_fit=parallel_chains_per_fit,
                progress_bar=progress_bar,
            )
            for job in posterior_jobs:
                print(
                    f"fit complete: {job.job_id}: {job.job_dir / 'posterior.npz'}",
                    file=sys.stderr,
                )
            print(f"all jobs complete: {base_job_dir}", file=sys.stderr)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return 1
    return 0
