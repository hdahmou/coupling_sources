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

from fabric import Connection

from coupling_sources.core.data import MCMCSpecs

LOCAL_HOSTS = {"local", "localhost", "127.0.0.1", "::1"}
DEFAULT_REMOTE_PYTHON = ".venv/bin/python"


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
        host = _host_with_optional_user(values)
        max_concurrent_fits = int(values.get("max_concurrent_fits", 0))
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
            port=None if port is None else int(port),
            env=_mapping(values.get("env"), "machine.env"),
            connect_kwargs=_mapping(values.get("connect_kwargs"), "machine.connect_kwargs"),
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
            raise ValueError("compute config can include at most one local machine entry")
        if not any(machine.active for machine in machines):
            raise ValueError("compute config must include at least one active machine")

        return cls(
            machines=machines,
            python=str(values.get("python", DEFAULT_REMOTE_PYTHON)),
            env=_mapping(values.get("env"), "env"),
            connect_kwargs=_mapping(values.get("connect_kwargs"), "connect_kwargs"),
        )

    @property
    def active_machines(self) -> tuple[Machine, ...]:
        return tuple(machine for machine in self.machines if machine.active)

    @property
    def active_remote_machines(self) -> tuple[Machine, ...]:
        return tuple(machine for machine in self.active_machines if not machine.is_local)


@dataclass(frozen=True)
class PosteriorJob:
    job_id: str
    job_dir: Path


@dataclass(frozen=True)
class MachineAssignment:
    machine: Machine
    jobs: tuple[PosteriorJob, ...]


@dataclass(frozen=True)
class HostResult:
    host: str
    job_count: int
    failed_jobs: tuple[str, ...]


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
    progress_bar: bool,
) -> None:
    if not jobs:
        return

    config = ComputeConfig.from_mapping(compute_config)
    assignments = _assign_jobs(jobs, config.active_machines)
    results: list[HostResult] = []

    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = [
            executor.submit(
                _run_assignment,
                assignment,
                config,
                run_dir,
                mcmc,
                progress_bar,
            )
            for assignment in assignments
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"host complete: {result.host}: "
                f"jobs={result.job_count} failed={len(result.failed_jobs)}",
                flush=True,
            )

    failed = [job_id for result in results for job_id in result.failed_jobs]
    if failed:
        raise RuntimeError(f"posterior fits failed for {len(failed)} job(s): {failed[:10]}")


def _host_with_optional_user(values: Mapping[str, object]) -> str:
    if "host" not in values:
        raise ValueError("machine entries must include host")
    host = str(values["host"])
    user = values.get("username", values.get("user"))
    if user is None or "@" in host:
        return host
    return f"{user}@{host}"


def _mapping(value: object, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _string_values(values: Mapping[str, object] | None) -> dict[str, str]:
    return {} if values is None else {str(key): str(value) for key, value in values.items()}


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
) -> list[MachineAssignment]:
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
        MachineAssignment(machine, tuple(machine_jobs))
        for machine, machine_jobs in zip(machines, assigned, strict=True)
        if machine_jobs
    ]


def _run_assignment(
    assignment: MachineAssignment,
    config: ComputeConfig,
    run_dir: Path,
    mcmc: MCMCSpecs,
    progress_bar: bool,
) -> HostResult:
    if assignment.machine.is_local:
        return _run_local_machine(
            assignment,
            config,
            run_dir,
            mcmc,
            progress_bar,
        )
    return _run_remote_machine(
        assignment,
        config,
        run_dir,
        mcmc,
        progress_bar,
    )


def _machine_env(machine: Machine, config: ComputeConfig) -> dict[str, str]:
    env = {**_string_values(config.env), **_string_values(machine.env)}
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _connection(machine: Machine, config: ComputeConfig) -> Connection:
    kwargs: dict[str, object] = {
        "connect_kwargs": {
            **(config.connect_kwargs or {}),
            **(machine.connect_kwargs or {}),
        }
    }
    if machine.port is not None:
        kwargs["port"] = machine.port
    return Connection(machine.host, **kwargs)


def _write_assignment(
    *,
    run_dir: Path,
    machine: Machine,
    jobs: Sequence[PosteriorJob],
    mcmc: MCMCSpecs,
    progress_bar: bool,
) -> Path:
    transfer_dir = run_dir / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = transfer_dir / f"assigned_jobs_{_safe_path_part(machine.host)}.json"
    assignment = {
        "worker_label": _safe_path_part(machine.host),
        "jobs": [job.job_id for job in jobs],
        "mcmc": asdict(mcmc),
        "progress_bar": progress_bar,
    }
    assignment_path.write_text(json.dumps(assignment, indent=2) + "\n")
    return assignment_path


def _make_input_archive(
    *,
    run_dir: Path,
    machine: Machine,
    jobs: Sequence[PosteriorJob],
    assignment_path: Path,
) -> Path:
    archive_path = run_dir / "transfer" / f"input_{_safe_path_part(machine.host)}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in (run_dir / "config.json", run_dir / "job_grid.csv", assignment_path):
            archive.add(path, arcname=path.relative_to(run_dir))
        for job in jobs:
            stimuli_path = job.job_dir / "stimuli.zarr"
            simulation_parameters_path = job.job_dir / "simulation_parameters.json"
            archive.add(stimuli_path, arcname=stimuli_path.relative_to(run_dir))
            archive.add(
                simulation_parameters_path,
                arcname=simulation_parameters_path.relative_to(run_dir),
            )
    return archive_path


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


def _shell_env(env: Mapping[str, str]) -> str:
    if not env:
        return ""
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + " "


def _remote_worker_command(
    *,
    machine: Machine,
    config: ComputeConfig,
    remote_run_dir: Path,
    remote_input_archive: Path,
    remote_assignment: Path,
    remote_result_archive: Path,
) -> str:
    args = [
        machine.remote_python_path(config.python),
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
    env = _shell_env(_machine_env(machine, config))
    return f"cd {shlex.quote(str(project_path))} && {env}{shlex.join(args)}"


def _run_remote_machine(
    assignment: MachineAssignment,
    config: ComputeConfig,
    run_dir: Path,
    mcmc: MCMCSpecs,
    progress_bar: bool,
) -> HostResult:
    machine = assignment.machine
    jobs = assignment.jobs
    assignment_path = _write_assignment(
        run_dir=run_dir,
        machine=machine,
        jobs=jobs,
        mcmc=mcmc,
        progress_bar=progress_bar,
    )
    input_archive = _make_input_archive(
        run_dir=run_dir,
        machine=machine,
        jobs=jobs,
        assignment_path=assignment_path,
    )

    remote_run_dir = machine.remote_project_path() / _project_relative_run_dir(run_dir)
    remote_transfer_dir = remote_run_dir / "transfer"
    remote_input_archive = remote_transfer_dir / input_archive.name
    remote_assignment = remote_run_dir / assignment_path.relative_to(run_dir)
    remote_result_archive = remote_transfer_dir / f"results_{_safe_path_part(machine.host)}.tar.gz"
    local_result_archive = run_dir / "transfer" / remote_result_archive.name
    command = _remote_worker_command(
        machine=machine,
        config=config,
        remote_run_dir=remote_run_dir,
        remote_input_archive=remote_input_archive,
        remote_assignment=remote_assignment,
        remote_result_archive=remote_result_archive,
    )

    print(f"remote host start: {machine.host}: jobs={len(jobs)}", flush=True)
    print(f"remote run dir: {machine.host}: {remote_run_dir}", flush=True)
    with _connection(machine, config) as connection:
        connection.run(f"mkdir -p {shlex.quote(str(remote_transfer_dir))}")
        connection.put(str(input_archive), remote=str(remote_input_archive))
        result = connection.run(command, warn=True)
        try:
            connection.get(str(remote_result_archive), local=str(local_result_archive))
        except Exception as exc:
            raise RuntimeError(
                f"failed to fetch result archive from {machine.host}: {remote_result_archive}"
            ) from exc

    _safe_extract(local_result_archive, run_dir)
    return _host_result(run_dir, machine.host, jobs, result.exited)


def _run_local_machine(
    assignment: MachineAssignment,
    config: ComputeConfig,
    run_dir: Path,
    mcmc: MCMCSpecs,
    progress_bar: bool,
) -> HostResult:
    from coupling_sources.compute.fabric_worker import run_assigned_jobs

    machine = assignment.machine
    jobs = assignment.jobs
    assignment_path = _write_assignment(
        run_dir=run_dir,
        machine=machine,
        jobs=jobs,
        mcmc=mcmc,
        progress_bar=progress_bar,
    )

    previous_env = os.environ.copy()
    os.environ.update(_machine_env(machine, config))
    try:
        exit_code = run_assigned_jobs(
            run_dir=run_dir,
            assigned_jobs_path=assignment_path,
            max_parallel=machine.max_concurrent_fits,
            result_archive_path=None,
        )
    finally:
        os.environ.clear()
        os.environ.update(previous_env)

    return _host_result(run_dir, machine.host, jobs, exit_code)


def _host_result(
    run_dir: Path, host: str, jobs: Sequence[PosteriorJob], exit_code: int
) -> HostResult:
    failed = _failed_jobs(run_dir, jobs)
    if exit_code != 0 and not failed:
        raise RuntimeError(f"host {host} exited with code {exit_code}")
    return HostResult(host=host, job_count=len(jobs), failed_jobs=tuple(failed))


def _failed_jobs(run_dir: Path, jobs: Sequence[PosteriorJob]) -> list[str]:
    failed: list[str] = []
    for job in jobs:
        status_path = run_dir / "job_status" / f"{_safe_path_part(job.job_id)}.json"
        if not status_path.exists():
            failed.append(job.job_id)
            continue
        status = json.loads(status_path.read_text())
        if not status.get("ok", False):
            failed.append(job.job_id)
    return failed
