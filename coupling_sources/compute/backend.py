from __future__ import annotations

import getpass
import json
import os
import re
import shlex
import tarfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event

from fabric import Connection

from coupling_sources.compute.archives import safe_extract_tar
from coupling_sources.core.data import MCMCSpecs

LOCAL_HOSTS = {"local", "localhost", "127.0.0.1", "::1"}
DEFAULT_REMOTE_PYTHON = ".venv/bin/python"
DEFAULT_JAX_COMPILATION_CACHE_DIR = ".jax_cache"


def _config_int(value: object, name: str) -> int:
    if not isinstance(value, int | float | str):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _config_mapping(value: object | None, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class Machine:
    host: str
    max_concurrent_fits: int
    project_path: Path | None = None
    python: str | None = None
    port: int | None = None
    env: dict[str, object] | None = None
    connect_kwargs: dict[str, object] | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "Machine":
        if "host" not in values:
            raise ValueError("machine entries must include host")
        host = str(values["host"])

        max_concurrent_fits = _config_int(
            values.get("max_concurrent_fits", 0),
            "max_concurrent_fits",
        )
        if max_concurrent_fits < 0:
            raise ValueError(f"max_concurrent_fits cannot be negative for {host}")

        project_path = values.get("project_path")
        python = values.get("python")
        port = values.get("port")
        return cls(
            host=host,
            max_concurrent_fits=max_concurrent_fits,
            project_path=None if project_path is None else Path(str(project_path)),
            python=None if python is None else str(python),
            port=None if port is None else _config_int(port, "port"),
            env=_config_mapping(values.get("env"), "env"),
            connect_kwargs=_config_mapping(
                values.get("connect_kwargs"),
                "connect_kwargs",
            ),
        )

    @property
    def active(self) -> bool:
        return self.max_concurrent_fits > 0

    @property
    def is_local(self) -> bool:
        host = self.host.rsplit("@", 1)[-1].rsplit(":", 1)[0]
        return host in LOCAL_HOSTS

    def remote_project_path(self) -> Path:
        if self.project_path is None:
            raise ValueError(f"remote machine {self.host!r} must define project_path")
        return self.project_path

    def remote_python_path(self, default_python: str) -> str:
        configured = self.python or default_python
        python_path = Path(configured)
        if python_path.is_absolute():
            return str(python_path)
        return str(self.remote_project_path() / python_path)


@dataclass(frozen=True)
class ComputeConfig:
    machines: tuple[Machine, ...]
    python: str = DEFAULT_REMOTE_PYTHON
    env: dict[str, object] | None = None
    connect_kwargs: dict[str, object] | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ComputeConfig":
        raw_machines = values.get("machines")
        if not isinstance(raw_machines, list) or not raw_machines:
            raise ValueError("compute config must include at least one machine")

        machines = tuple(
            Machine.from_mapping(machine)
            for machine in raw_machines
            if isinstance(machine, Mapping)
        )
        if len(machines) != len(raw_machines):
            raise ValueError("compute config machines must be objects")
        if sum(machine.is_local for machine in machines) > 1:
            raise ValueError(
                "compute config can include at most one local machine entry"
            )
        if not any(machine.active for machine in machines):
            raise ValueError("compute config must include at least one active machine")

        return cls(
            machines=machines,
            python=str(values.get("python", DEFAULT_REMOTE_PYTHON)),
            env=_config_mapping(values.get("env"), "env"),
            connect_kwargs=_config_mapping(
                values.get("connect_kwargs"),
                "connect_kwargs",
            ),
        )

    @property
    def active_machines(self) -> tuple[Machine, ...]:
        return tuple(machine for machine in self.machines if machine.active)

    @property
    def active_remote_machines(self) -> tuple[Machine, ...]:
        return tuple(
            machine for machine in self.active_machines if not machine.is_local
        )


@dataclass(frozen=True)
class PosteriorJob:
    job_id: str
    job_dir: Path


@dataclass(frozen=True)
class HostResult:
    host: str
    job_count: int
    failed_jobs: tuple[str, ...]

    @classmethod
    def from_jobs(
        cls,
        *,
        host: str,
        jobs: Sequence[PosteriorJob],
        exit_code: int,
    ) -> "HostResult":
        failed: list[str] = []
        for job in jobs:
            status_path = job.job_dir / "status.json"
            if not status_path.exists():
                failed.append(job.job_id)
                continue

            status = json.loads(status_path.read_text())
            if not status.get("ok", False):
                failed.append(job.job_id)

        if exit_code != 0 and not failed:
            raise RuntimeError(f"host {host} exited with code {exit_code}")
        return cls(host=host, job_count=len(jobs), failed_jobs=tuple(failed))


def add_prompted_ssh_password(compute_config: dict[str, object]) -> None:
    config = ComputeConfig.from_mapping(compute_config)
    if not config.active_remote_machines:
        return

    connect_kwargs = compute_config.setdefault("connect_kwargs", {})
    if not isinstance(connect_kwargs, dict):
        raise ValueError("connect_kwargs must be an object")
    if "password" not in connect_kwargs:
        connect_kwargs["password"] = getpass.getpass("SSH password: ")


def run_posterior_jobs(
    *,
    compute_config: dict[str, object],
    run_dir: Path,
    jobs: Sequence[PosteriorJob],
    mcmc: MCMCSpecs,
    parallel_chains_per_fit: int | None = None,
    progress_bar: bool,
) -> None:
    if not jobs:
        return

    config = ComputeConfig.from_mapping(compute_config)
    assignments = _assign_jobs(jobs, config.active_machines)
    results: list[HostResult] = []
    shutdown_event = Event()
    runner = _PosteriorJobRunner(
        config=config,
        run_dir=run_dir,
        mcmc=mcmc,
        parallel_chains_per_fit=parallel_chains_per_fit,
        progress_bar=progress_bar,
        shutdown_event=shutdown_event,
    )

    executor = ThreadPoolExecutor(max_workers=len(assignments))
    futures = []
    try:
        futures = [
            executor.submit(runner.run, machine, machine_jobs)
            for machine, machine_jobs in assignments
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"host complete: {result.host}: "
                f"jobs={result.job_count} failed={len(result.failed_jobs)}",
                flush=True,
            )
    except BaseException:
        shutdown_event.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    failed = [job_id for result in results for job_id in result.failed_jobs]
    if failed:
        raise RuntimeError(
            f"posterior fits failed for {len(failed)} job(s): {failed[:10]}"
        )


def _safe_path_part(value: object) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return name or "machine"


def _project_relative_run_dir(run_dir: Path) -> Path:
    resolved = run_dir.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return Path("artifacts") / resolved.name


def _assign_jobs(
    jobs: Sequence[PosteriorJob], machines: Sequence[Machine]
) -> list[tuple[Machine, tuple[PosteriorJob, ...]]]:
    slots = [
        machine_index
        for machine_index, machine in enumerate(machines)
        for _ in range(machine.max_concurrent_fits)
    ]
    if not slots:
        raise ValueError("no active machine slots configured")

    assigned: list[list[PosteriorJob]] = [[] for _ in machines]
    for job_index, job in enumerate(jobs):
        assigned[slots[job_index % len(slots)]].append(job)

    return [
        (machine, tuple(machine_jobs))
        for machine, machine_jobs in zip(machines, assigned, strict=True)
        if machine_jobs
    ]


def _xla_flags_with_host_device_count(flags: object | None, count: int) -> str:
    if count < 1:
        raise ValueError("parallel_chains_per_fit must be positive")

    desired = f"--xla_force_host_platform_device_count={count}"
    if flags is None or str(flags).strip() == "":
        return desired

    text = str(flags)
    pattern = r"(^|\s)--xla_force_host_platform_device_count=\S+"
    if re.search(pattern, text):
        return re.sub(pattern, lambda match: f"{match.group(1)}{desired}", text)
    return f"{text} {desired}"


def _machine_env(
    machine: Machine,
    config: ComputeConfig,
    parallel_chains_per_fit: int | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    for values in (config.env, machine.env):
        if values is not None:
            env.update({str(key): str(value) for key, value in values.items()})
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "true")
    env.setdefault("JAX_COMPILATION_CACHE_DIR", DEFAULT_JAX_COMPILATION_CACHE_DIR)
    if parallel_chains_per_fit is not None:
        env["XLA_FLAGS"] = _xla_flags_with_host_device_count(
            env.get("XLA_FLAGS"),
            parallel_chains_per_fit,
        )
    return env


@dataclass(frozen=True)
class _PosteriorJobRunner:
    config: ComputeConfig
    run_dir: Path
    mcmc: MCMCSpecs
    parallel_chains_per_fit: int | None
    progress_bar: bool
    shutdown_event: Event

    def run(self, machine: Machine, jobs: tuple[PosteriorJob, ...]) -> HostResult:
        if machine.is_local:
            return self._run_local(machine, jobs)
        return self._run_remote(machine, jobs)

    def _write_assignment(
        self,
        machine: Machine,
        jobs: Sequence[PosteriorJob],
    ) -> Path:
        transfer_dir = self.run_dir / "transfer"
        transfer_dir.mkdir(parents=True, exist_ok=True)
        assignment_path = (
            transfer_dir / f"assigned_jobs_{_safe_path_part(machine.host)}.json"
        )
        assigned_jobs = {
            "worker_label": _safe_path_part(machine.host),
            "jobs": [job.job_id for job in jobs],
            "mcmc": asdict(self.mcmc),
            "progress_bar": self.progress_bar,
        }
        assignment_path.write_text(json.dumps(assigned_jobs, indent=2) + "\n")
        return assignment_path

    def _make_input_archive(
        self,
        machine: Machine,
        jobs: Sequence[PosteriorJob],
        assignment_path: Path,
    ) -> Path:
        archive_path = (
            self.run_dir / "transfer" / f"input_{_safe_path_part(machine.host)}.tar.gz"
        )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in (
                self.run_dir / "config.json",
                self.run_dir / "job_grid.csv",
                assignment_path,
            ):
                archive.add(path, arcname=path.relative_to(self.run_dir))
            for job in jobs:
                job_config_path = job.job_dir / "job.json"
                archive.add(
                    job_config_path,
                    arcname=job_config_path.relative_to(self.run_dir),
                )
        return archive_path

    def _connection(self, machine: Machine) -> Connection:
        kwargs: dict[str, object] = {
            "connect_kwargs": {
                **(self.config.connect_kwargs or {}),
                **(machine.connect_kwargs or {}),
            }
        }
        if machine.port is not None:
            kwargs["port"] = machine.port
        return Connection(machine.host, **kwargs)

    def _remote_worker_command(
        self,
        *,
        machine: Machine,
        remote_run_dir: Path,
        remote_input_archive: Path,
        remote_assignment: Path,
        remote_result_archive: Path,
    ) -> str:
        args = [
            machine.remote_python_path(self.config.python),
            "-m",
            "coupling_sources.compute.fabric_worker",
            "--run-dir",
            str(remote_run_dir),
            "--input-archive",
            str(remote_input_archive),
            "--assigned-jobs",
            str(remote_assignment),
            "--max-parallel",
            str(machine.max_concurrent_fits),
            "--result-archive",
            str(remote_result_archive),
        ]
        project_path = machine.remote_project_path()
        env = _machine_env(machine, self.config, self.parallel_chains_per_fit)
        env_prefix = (
            " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + " "
            if env
            else ""
        )
        return f"cd {shlex.quote(str(project_path))} && {env_prefix}{shlex.join(args)}"

    def _run_remote(
        self,
        machine: Machine,
        jobs: tuple[PosteriorJob, ...],
    ) -> HostResult:
        assignment_path = self._write_assignment(machine, jobs)
        input_archive = self._make_input_archive(machine, jobs, assignment_path)

        remote_run_dir = machine.remote_project_path() / _project_relative_run_dir(
            self.run_dir
        )
        remote_transfer_dir = remote_run_dir / "transfer"
        remote_input_archive = remote_transfer_dir / input_archive.name
        remote_assignment = remote_run_dir / assignment_path.relative_to(self.run_dir)
        remote_result_archive = (
            remote_transfer_dir / f"results_{_safe_path_part(machine.host)}.tar.gz"
        )
        local_result_archive = self.run_dir / "transfer" / remote_result_archive.name
        command = self._remote_worker_command(
            machine=machine,
            remote_run_dir=remote_run_dir,
            remote_input_archive=remote_input_archive,
            remote_assignment=remote_assignment,
            remote_result_archive=remote_result_archive,
        )

        print(f"remote host start: {machine.host}: jobs={len(jobs)}", flush=True)
        print(f"remote run dir: {machine.host}: {remote_run_dir}", flush=True)
        with self._connection(machine) as connection:
            connection.run(f"mkdir -p {shlex.quote(str(remote_transfer_dir))}")
            connection.put(str(input_archive), remote=str(remote_input_archive))
            result = connection.run(command, warn=True)
            try:
                connection.get(
                    str(remote_result_archive),
                    local=str(local_result_archive),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"failed to fetch result archive from {machine.host}: "
                    f"{remote_result_archive}"
                ) from exc

        safe_extract_tar(local_result_archive, self.run_dir)
        return HostResult.from_jobs(
            host=machine.host,
            jobs=jobs,
            exit_code=result.exited,
        )

    def _run_local(
        self,
        machine: Machine,
        jobs: tuple[PosteriorJob, ...],
    ) -> HostResult:
        assignment_path = self._write_assignment(machine, jobs)

        previous_env = os.environ.copy()
        os.environ.update(
            _machine_env(machine, self.config, self.parallel_chains_per_fit)
        )
        try:
            from coupling_sources.compute.fabric_worker import run_assigned_jobs

            exit_code = run_assigned_jobs(
                run_dir=self.run_dir,
                assigned_jobs_path=assignment_path,
                max_parallel=machine.max_concurrent_fits,
                result_archive_path=None,
                shutdown_event=self.shutdown_event,
            )
        finally:
            os.environ.clear()
            os.environ.update(previous_env)

        return HostResult.from_jobs(
            host=machine.host,
            jobs=jobs,
            exit_code=exit_code,
        )
