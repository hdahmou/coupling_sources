from __future__ import annotations

import getpass
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from distributed import Client, LocalCluster, Nanny, Scheduler, as_completed
from distributed.deploy.spec import SpecCluster
from distributed.deploy.ssh import Scheduler as SSHScheduler
from distributed.deploy.ssh import Worker as SSHWorker


LOCAL_HOSTS = {"local", "localhost", "127.0.0.1", "::1"}
DEFAULT_REMOTE_PYTHON = ".venv/bin/python"
DEFAULT_TUNNEL_WORKER_PORT_START = 39000
DEFAULT_TUNNEL_NANNY_PORT_START = 40000

T = TypeVar("T")


def _split_host(value: object) -> tuple[str | None, str]:
    host = str(value)
    if "@" not in host:
        return None, host
    user, address = host.rsplit("@", 1)
    return user or None, address


def _object_mapping(value: object, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True)
class Machine:
    host: str
    username: str | None
    max_concurrent_fits: int
    threads_per_worker: int
    project_path: Path | None
    python: str | None
    port: int | None
    env: dict[str, object]
    connect_options: dict[str, object]
    worker_options: dict[str, object]

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, object], *, default_worker_count: int | None = None
    ) -> "Machine":
        if "host" not in values:
            raise ValueError("machine entries must include host")

        shorthand_user, host = _split_host(values["host"])
        configured_user = values.get("username", values.get("user", shorthand_user))
        max_concurrent_fits = int(values.get("max_concurrent_fits", default_worker_count or 0))
        threads_per_worker = int(values.get("threads_per_worker", 1))
        if max_concurrent_fits < 0:
            raise ValueError(f"max_concurrent_fits cannot be negative for {host}")
        if threads_per_worker < 1:
            raise ValueError(f"threads_per_worker must be positive for {host}")

        project_path = values.get("project_path")
        port = values.get("port")
        python = values.get("python")
        return cls(
            host=host,
            username=None if configured_user is None else str(configured_user),
            max_concurrent_fits=max_concurrent_fits,
            threads_per_worker=threads_per_worker,
            project_path=None if project_path is None else Path(str(project_path)),
            python=None if python is None else str(python),
            port=None if port is None else int(port),
            env=_object_mapping(values.get("env"), "machine.env"),
            connect_options=_object_mapping(
                values.get("connect_options"), "machine.connect_options"
            ),
            worker_options=_object_mapping(values.get("worker_options"), "machine.worker_options"),
        )

    @property
    def is_local(self) -> bool:
        return self.host in LOCAL_HOSTS

    def as_mapping(self) -> dict[str, object]:
        values: dict[str, object] = {
            "host": self.host,
            "max_concurrent_fits": self.max_concurrent_fits,
            "threads_per_worker": self.threads_per_worker,
        }
        if self.username is not None:
            values["username"] = self.username
        if self.project_path is not None:
            values["project_path"] = str(self.project_path)
        if self.python is not None:
            values["python"] = self.python
        if self.port is not None:
            values["port"] = self.port
        if self.env:
            values["env"] = dict(self.env)
        if self.connect_options:
            values["connect_options"] = dict(self.connect_options)
        if self.worker_options:
            values["worker_options"] = dict(self.worker_options)
        return values

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

    def ssh_connect_options(self, config: "ComputeConfig") -> dict[str, object]:
        options = {**config.connect_options, **self.connect_options}
        if self.username is not None:
            options["username"] = self.username
        if self.port is not None:
            options["port"] = self.port
        return options

    def env_with(self, config: "ComputeConfig") -> dict[str, str]:
        return {**_string_values(config.env), **_string_values(self.env)}


def _string_values(values: Mapping[str, object]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items()}


@dataclass(frozen=True)
class ComputeConfig:
    machines: tuple[Machine, ...]
    python: str
    connect_options: dict[str, object]
    env: dict[str, object]
    worker_options: dict[str, object]
    scheduler_options: dict[str, object]
    tunnel: dict[str, object]
    scheduler: dict[str, object] | None
    scheduler_host: str | None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ComputeConfig":
        raw_machines = values.get("machines")
        if not isinstance(raw_machines, list) or not raw_machines:
            raise ValueError("compute config must include at least one machine")
        if not all(isinstance(machine, dict) for machine in raw_machines):
            raise ValueError("compute config machines must be objects")

        machines = tuple(Machine.from_mapping(machine) for machine in raw_machines)
        if sum(machine.is_local for machine in machines) > 1:
            raise ValueError("compute config can include at most one local machine entry")
        if not any(machine.max_concurrent_fits > 0 for machine in machines):
            raise ValueError(
                "compute config must include at least one machine with "
                "max_concurrent_fits > 0"
            )

        scheduler = values.get("scheduler")
        return cls(
            machines=machines,
            python=str(values.get("python", values.get("remote_python", DEFAULT_REMOTE_PYTHON))),
            connect_options=_object_mapping(values.get("connect_options"), "connect_options"),
            env=_object_mapping(values.get("env"), "env"),
            worker_options=_object_mapping(values.get("worker_options"), "worker_options"),
            scheduler_options=_object_mapping(
                values.get("scheduler_options"), "scheduler_options"
            ),
            tunnel=_object_mapping(values.get("tunnel"), "tunnel"),
            scheduler=None if scheduler is None else _object_mapping(scheduler, "scheduler"),
            scheduler_host=(
                None if values.get("scheduler_host") is None else str(values["scheduler_host"])
            ),
        )

    @property
    def local_machine(self) -> Machine | None:
        return next((machine for machine in self.machines if machine.is_local), None)

    @property
    def remote_machines(self) -> tuple[Machine, ...]:
        return tuple(machine for machine in self.machines if not machine.is_local)

    @property
    def worker_machines(self) -> tuple[Machine, ...]:
        return tuple(machine for machine in self.machines if machine.max_concurrent_fits > 0)

    @property
    def remote_worker_machines(self) -> tuple[Machine, ...]:
        return tuple(machine for machine in self.worker_machines if not machine.is_local)

    @property
    def tunnel_enabled(self) -> bool:
        configured = self.tunnel.get("enabled")
        if configured is None:
            return self.scheduler_is_local and bool(self.remote_worker_machines)
        if isinstance(configured, str):
            return configured.strip().lower() not in {"0", "false", "no", "off"}
        return bool(configured)

    @property
    def tunnel_worker_port_start(self) -> int:
        return _port_value(
            self.tunnel.get("worker_port_start", DEFAULT_TUNNEL_WORKER_PORT_START),
            "tunnel.worker_port_start",
        )

    @property
    def tunnel_nanny_port_start(self) -> int:
        return _port_value(
            self.tunnel.get("nanny_port_start", DEFAULT_TUNNEL_NANNY_PORT_START),
            "tunnel.nanny_port_start",
        )

    @property
    def uses_ssh(self) -> bool:
        return bool(self.remote_worker_machines) or self.scheduler_uses_ssh

    @property
    def scheduler_uses_ssh(self) -> bool:
        if self.scheduler is None and self.scheduler_host is None:
            return False
        return not self.scheduler_is_local

    @property
    def scheduler_is_local(self) -> bool:
        if self.scheduler is not None:
            if "host" not in self.scheduler:
                raise ValueError("scheduler must include host")
            _, host = _split_host(self.scheduler["host"])
            return host in LOCAL_HOSTS

        if self.scheduler_host is not None:
            _, host = _split_host(self.scheduler_host)
            return host in LOCAL_HOSTS

        return False

    def scheduler_machine(self) -> Machine:
        if self.scheduler is not None:
            if "host" not in self.scheduler:
                raise ValueError("scheduler must include host")
            _, host = _split_host(self.scheduler["host"])
            if host in LOCAL_HOSTS:
                raise ValueError("local scheduler does not have a remote machine")
            base = next((machine for machine in self.remote_machines if machine.host == host), None)
            values = {"host": self.scheduler["host"], **self.scheduler}
            if base is not None:
                values = {**base.as_mapping(), **values}
            return Machine.from_mapping(values, default_worker_count=1)

        if self.scheduler_host is not None:
            _, host = _split_host(self.scheduler_host)
            if host in LOCAL_HOSTS:
                raise ValueError("local scheduler does not have a remote machine")
            for machine in self.remote_machines:
                if machine.host == host:
                    return machine
            raise ValueError(f"scheduler_host {host!r} is not listed in machines")

        if not self.remote_worker_machines:
            raise ValueError("SSH compute requires at least one active non-local machine")
        return self.remote_worker_machines[0]


def add_prompted_ssh_password(compute_config: dict[str, object]) -> None:
    if not ComputeConfig.from_mapping(compute_config).uses_ssh:
        return

    options = compute_config.setdefault("connect_options", {})
    if not isinstance(options, dict):
        raise ValueError("connect_options must be an object when prompting for an SSH password")
    options["password"] = getpass.getpass("SSH password: ")


def _port_value(value: object, name: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return port


def _safe_path_part(value: object) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return name or "machine"


def _run_dir_relative_to_project(log_path: Path) -> Path:
    run_dir = log_path.parent.resolve()
    try:
        return run_dir.relative_to(Path.cwd().resolve())
    except ValueError:
        return Path("artifacts") / run_dir.name


def _remote_worker_log_dir(machine: Machine, log_path: Path) -> Path:
    return (
        machine.remote_project_path()
        / _run_dir_relative_to_project(log_path)
        / "worker_logs"
        / _safe_path_part(machine.host)
    )


def _remote_worker_env(
    machine: Machine, config: ComputeConfig, log_path: Path
) -> dict[str, str]:
    project_path = machine.remote_project_path()
    project_pythonpath = f"{project_path}:{project_path / 'scripts'}"
    env = machine.env_with(config)
    configured_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        project_pythonpath
        if configured_pythonpath is None
        else f"{project_pythonpath}:{configured_pythonpath}"
    )
    env["COUPLING_SOURCES_WORKER_LOG_DIR"] = str(_remote_worker_log_dir(machine, log_path))
    env["COUPLING_SOURCES_WORKER_HOST"] = machine.host
    return env


def _local_worker_env(machine: Machine, config: ComputeConfig, log_path: Path) -> dict[str, str]:
    env = machine.env_with(config)
    env["COUPLING_SOURCES_RUN_LOG"] = str(log_path)
    env["COUPLING_SOURCES_WORKER_HOST"] = machine.host
    return env


def _merge_worker_options(
    kwargs: dict[str, Any], machine: Machine, config: ComputeConfig
) -> dict[str, object]:
    merged = dict(kwargs)
    merged["env"] = dict(kwargs.get("env", {}))
    for configured in (config.worker_options, machine.worker_options):
        options = dict(configured)
        env = options.pop("env", {})
        if env is None:
            env = {}
        if not isinstance(env, dict):
            raise ValueError("worker_options.env must be an object")
        merged.update(options)
        merged["env"].update(_string_values(env))
    return merged


def _remote_worker_slot_count(config: ComputeConfig) -> int:
    return sum(machine.max_concurrent_fits for machine in config.remote_worker_machines)


def _validate_tunnel_ports(config: ComputeConfig) -> None:
    if not config.scheduler_is_local:
        raise ValueError("tunnel mode requires a local scheduler")

    count = _remote_worker_slot_count(config)
    if count == 0:
        return

    worker_start = config.tunnel_worker_port_start
    nanny_start = config.tunnel_nanny_port_start
    worker_end = worker_start + count - 1
    nanny_end = nanny_start + count - 1
    if worker_end > 65535:
        raise ValueError("tunnel.worker_port_start leaves too few available ports")
    if nanny_end > 65535:
        raise ValueError("tunnel.nanny_port_start leaves too few available ports")
    if max(worker_start, nanny_start) <= min(worker_end, nanny_end):
        raise ValueError("tunnel worker and nanny port ranges must not overlap")

    scheduler_port = config.scheduler_options.get("port")
    if scheduler_port is not None:
        scheduler_port = _port_value(scheduler_port, "scheduler_options.port")
        if worker_start <= scheduler_port <= worker_end:
            raise ValueError("scheduler port overlaps tunnel worker port range")
        if nanny_start <= scheduler_port <= nanny_end:
            raise ValueError("scheduler port overlaps tunnel nanny port range")


def _tunnel_worker_port(config: ComputeConfig, remote_slot_index: int) -> int:
    return config.tunnel_worker_port_start + remote_slot_index


def _tunnel_nanny_port(config: ComputeConfig, remote_slot_index: int) -> int:
    return config.tunnel_nanny_port_start + remote_slot_index


def _ssh_worker_kwargs(
    machine: Machine,
    config: ComputeConfig,
    log_path: Path,
    *,
    remote_slot_index: int | None = None,
) -> dict[str, object]:
    options = _merge_worker_options(
        {
            "nthreads": machine.threads_per_worker,
            "env": _remote_worker_env(machine, config, log_path),
        },
        machine,
        config,
    )
    if remote_slot_index is None:
        options["n_workers"] = machine.max_concurrent_fits
        return options

    worker_port = _tunnel_worker_port(config, remote_slot_index)
    nanny_port = _tunnel_nanny_port(config, remote_slot_index)
    options.pop("n_workers", None)
    options.update(
        {
            "host": "127.0.0.1",
            "port": nanny_port,
            "worker_port": worker_port,
            "contact_address": f"tcp://127.0.0.1:{worker_port}",
        }
    )
    return options


def _local_worker_kwargs(
    machine: Machine, config: ComputeConfig, log_path: Path
) -> dict[str, object]:
    return _merge_worker_options(
        {
            "nthreads": machine.threads_per_worker,
            "env": _local_worker_env(machine, config, log_path),
            "preload": [str(Path(__file__).with_name("dask_log_preload.py"))],
        },
        machine,
        config,
    )


def _local_cluster_options(
    machine: Machine, config: ComputeConfig, log_path: Path
) -> dict[str, object]:
    options = _local_worker_kwargs(machine, config, log_path)
    options["threads_per_worker"] = options.pop("nthreads")
    return options


def _scheduler_spec(config: ComputeConfig) -> dict[str, object]:
    if config.scheduler_is_local:
        return {"cls": Scheduler, "options": dict(config.scheduler_options)}

    scheduler_machine = config.scheduler_machine()
    return {
        "cls": SSHScheduler,
        "options": {
            "address": scheduler_machine.host,
            "connect_options": scheduler_machine.ssh_connect_options(config),
            "kwargs": dict(config.scheduler_options),
            "remote_python": scheduler_machine.remote_python_path(config.python),
        },
    }


def _build_ssh_cluster(config: ComputeConfig, log_path: Path) -> SpecCluster:
    scheduler = _scheduler_spec(config)

    workers: dict[str, dict[str, object]] = {}
    if config.tunnel_enabled:
        _validate_tunnel_ports(config)
        remote_slot_index = 0
        for machine in config.remote_worker_machines:
            print(
                f"remote worker log dir: {machine.host}: {_remote_worker_log_dir(machine, log_path)}",
                file=sys.stderr,
            )
            for slot in range(machine.max_concurrent_fits):
                worker_port = _tunnel_worker_port(config, remote_slot_index)
                nanny_port = _tunnel_nanny_port(config, remote_slot_index)
                print(
                    "remote worker tunnel ports: "
                    f"{machine.host} slot={slot} "
                    f"worker=127.0.0.1:{worker_port} "
                    f"nanny=127.0.0.1:{nanny_port}",
                    file=sys.stderr,
                )
                workers[f"{machine.host}-{remote_slot_index}"] = {
                    "cls": SSHWorker,
                    "options": {
                        "address": machine.host,
                        "connect_options": machine.ssh_connect_options(config),
                        "kwargs": _ssh_worker_kwargs(
                            machine,
                            config,
                            log_path,
                            remote_slot_index=remote_slot_index,
                        ),
                        "worker_class": "distributed.Nanny",
                        "remote_python": machine.remote_python_path(config.python),
                    },
                }
                remote_slot_index += 1
    else:
        for index, machine in enumerate(config.remote_worker_machines):
            print(
                f"remote worker log dir: {machine.host}: {_remote_worker_log_dir(machine, log_path)}",
                file=sys.stderr,
            )
            workers[f"{machine.host}-{index}"] = {
                "cls": SSHWorker,
                "options": {
                    "address": machine.host,
                    "connect_options": machine.ssh_connect_options(config),
                    "kwargs": _ssh_worker_kwargs(machine, config, log_path),
                    "worker_class": "distributed.Nanny",
                    "remote_python": machine.remote_python_path(config.python),
                },
            }

    local_machine = config.local_machine
    if local_machine is not None:
        for slot in range(local_machine.max_concurrent_fits):
            workers[f"{local_machine.host}-{slot}"] = {
                "cls": Nanny,
                "options": _local_worker_kwargs(local_machine, config, log_path),
            }

    return SpecCluster(workers, scheduler, name="CouplingSourcesSSHCluster")


@contextmanager
def _compute_client(compute_config: dict[str, object], log_path: Path) -> Iterator[Client]:
    config = ComputeConfig.from_mapping(compute_config)
    local_machine = config.local_machine
    if (
        local_machine is not None
        and local_machine.max_concurrent_fits > 0
        and config.worker_machines == (local_machine,)
    ):
        with LocalCluster(
            n_workers=local_machine.max_concurrent_fits,
            **_local_cluster_options(local_machine, config, log_path),
        ) as cluster, Client(cluster) as client:
            yield client
        return

    with _build_ssh_cluster(config, log_path) as cluster, Client(cluster) as client:
        yield client


def run_tasks(
    compute_config: dict[str, object],
    log_path: Path,
    task: Callable[..., object],
    submissions: Iterable[tuple[T, dict[str, object]]],
) -> Iterator[tuple[T, object]]:
    with _compute_client(compute_config, log_path) as client:
        items = list(submissions)
        worker_count = len(client.scheduler_info().get("workers", {}))
        print(
            f"cluster ready: {worker_count} worker(s); submitting {len(items)} task(s)",
            file=sys.stderr,
        )
        futures = {client.submit(task, **kwargs): metadata for metadata, kwargs in items}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            print(f"task result received: {completed}/{len(futures)}", file=sys.stderr)
            yield futures[future], result
