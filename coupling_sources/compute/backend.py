from __future__ import annotations

import getpass
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from distributed import Client, LocalCluster, as_completed
from distributed.deploy.spec import SpecCluster
from distributed.deploy.ssh import Scheduler as SSHScheduler
from distributed.deploy.ssh import Worker as SSHWorker


LOCAL_HOSTS = {"local", "localhost", "127.0.0.1", "::1"}
DEFAULT_REMOTE_PYTHON = ".venv/bin/python"

T = TypeVar("T")


def _compute_machines(compute_config: dict[str, object]) -> list[dict[str, object]]:
    machines = compute_config.get("machines")
    if not isinstance(machines, list) or not machines:
        raise ValueError("compute config must include at least one machine")
    return machines


def _machine_host(machine: dict[str, object]) -> str:
    return str(machine["host"])


def _is_local_host(host: str) -> bool:
    return host in LOCAL_HOSTS


def _uses_ssh_compute(compute_config: dict[str, object]) -> bool:
    return any(
        not _is_local_host(_machine_host(machine))
        for machine in _compute_machines(compute_config)
    )


def add_prompted_ssh_password(compute_config: dict[str, object]) -> None:
    if not _uses_ssh_compute(compute_config):
        return

    options = compute_config.setdefault("connect_options", {})
    if not isinstance(options, dict):
        raise ValueError("connect_options must be an object when prompting for an SSH password")
    options["password"] = getpass.getpass("SSH password: ")


def _machine_worker_count(machine: dict[str, object]) -> int:
    count = int(machine["max_concurrent_fits"])
    if count < 1:
        raise ValueError(f"max_concurrent_fits must be positive for {_machine_host(machine)}")
    return count


def _machine_project_path(machine: dict[str, object]) -> Path:
    project_path = machine.get("project_path")
    if project_path is None:
        raise ValueError(f"remote machine {_machine_host(machine)!r} must define project_path")
    return Path(str(project_path))


def _remote_python_path(machine: dict[str, object], compute_config: dict[str, object]) -> str:
    configured = machine.get(
        "python",
        compute_config.get("python", compute_config.get("remote_python", DEFAULT_REMOTE_PYTHON)),
    )
    python_path = Path(str(configured))
    if python_path.is_absolute():
        return str(python_path)
    return str(_machine_project_path(machine) / python_path)


def _connect_options(
    machine: dict[str, object], compute_config: dict[str, object]
) -> dict[str, object]:
    options: dict[str, object] = {}
    global_options = compute_config.get("connect_options", {})
    machine_options = machine.get("connect_options", {})
    if isinstance(global_options, dict):
        options.update(global_options)
    if isinstance(machine_options, dict):
        options.update(machine_options)
    if "user" in machine:
        options["username"] = machine["user"]
    if "username" in machine:
        options["username"] = machine["username"]
    if "port" in machine:
        options["port"] = int(machine["port"])
    return options


def _remote_worker_env(
    machine: dict[str, object], compute_config: dict[str, object]
) -> dict[str, str]:
    project_path = _machine_project_path(machine)
    project_pythonpath = f"{project_path}:{project_path / 'scripts'}"
    env: dict[str, str] = {}

    for configured_env in (compute_config.get("env", {}), machine.get("env", {})):
        if isinstance(configured_env, dict):
            env.update({str(key): str(value) for key, value in configured_env.items()})

    configured_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        project_pythonpath
        if configured_pythonpath is None
        else f"{project_pythonpath}:{configured_pythonpath}"
    )
    return env


def _scheduler_machine_for(
    machines: list[dict[str, object]], compute_config: dict[str, object]
) -> dict[str, object]:
    scheduler = compute_config.get("scheduler")
    if isinstance(scheduler, dict):
        host = str(scheduler["host"])
        base = next((machine for machine in machines if _machine_host(machine) == host), {})
        return {**base, **scheduler}

    scheduler_host = compute_config.get("scheduler_host")
    if scheduler_host is not None:
        host = str(scheduler_host)
        for machine in machines:
            if _machine_host(machine) == host:
                return machine
        raise ValueError(f"scheduler_host {host!r} is not listed in machines")

    return next(machine for machine in machines if not _is_local_host(_machine_host(machine)))


def _ssh_worker_kwargs(
    machine: dict[str, object], compute_config: dict[str, object]
) -> dict[str, object]:
    kwargs: dict[str, Any] = {
        "nthreads": int(machine.get("threads_per_worker", 1)),
        "env": _remote_worker_env(machine, compute_config),
    }
    for configured in (compute_config.get("worker_options", {}), machine.get("worker_options", {})):
        if isinstance(configured, dict):
            configured = dict(configured)
            configured_env = configured.pop("env", {})
            kwargs.update(configured)
            if isinstance(configured_env, dict):
                kwargs["env"].update({str(key): str(value) for key, value in configured_env.items()})
    return kwargs


def _build_ssh_cluster(compute_config: dict[str, object]) -> SpecCluster:
    machines = _compute_machines(compute_config)
    local_machines = [machine for machine in machines if _is_local_host(_machine_host(machine))]
    remote_machines = [machine for machine in machines if not _is_local_host(_machine_host(machine))]
    if local_machines:
        raise ValueError(
            "local machines cannot be mixed with SSH machines; use an SSH-reachable "
            "hostname for the local machine, or keep the config local-only"
        )

    scheduler_machine = _scheduler_machine_for(remote_machines, compute_config)
    scheduler = {
        "cls": SSHScheduler,
        "options": {
            "address": _machine_host(scheduler_machine),
            "connect_options": _connect_options(scheduler_machine, compute_config),
            "kwargs": dict(compute_config.get("scheduler_options", {})),
            "remote_python": _remote_python_path(scheduler_machine, compute_config),
        },
    }

    workers: dict[str, dict[str, object]] = {}
    for machine in remote_machines:
        host = _machine_host(machine)
        for slot in range(_machine_worker_count(machine)):
            workers[f"{host}-{slot}"] = {
                "cls": SSHWorker,
                "options": {
                    "address": host,
                    "connect_options": _connect_options(machine, compute_config),
                    "kwargs": _ssh_worker_kwargs(machine, compute_config),
                    "worker_class": "distributed.Nanny",
                    "remote_python": _remote_python_path(machine, compute_config),
                },
            }

    return SpecCluster(workers, scheduler, name="CouplingSourcesSSHCluster")


@contextmanager
def _compute_client(
    compute_config: dict[str, object], log_path: Path
) -> Iterator[Client]:
    machines = _compute_machines(compute_config)
    if all(_is_local_host(_machine_host(machine)) for machine in machines):
        n_workers = sum(_machine_worker_count(machine) for machine in machines)
        with LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            env={"COUPLING_SOURCES_RUN_LOG": str(log_path)},
            preload=[str(Path(__file__).with_name("dask_log_preload.py"))],
        ) as cluster, Client(cluster) as client:
            yield client
        return

    with _build_ssh_cluster(compute_config) as cluster, Client(cluster) as client:
        yield client


def run_tasks(
    compute_config: dict[str, object],
    log_path: Path,
    task: Callable[..., object],
    submissions: Iterable[tuple[T, dict[str, object]]],
) -> Iterator[tuple[T, object]]:
    with _compute_client(compute_config, log_path) as client:
        futures = {
            client.submit(task, **kwargs): metadata
            for metadata, kwargs in submissions
        }
        for future in as_completed(futures):
            yield futures[future], future.result()
