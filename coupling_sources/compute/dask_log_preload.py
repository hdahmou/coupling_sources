from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def _safe_log_name(value: object) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return name or f"worker_{os.getpid()}"


def _worker_log_path(worker: object, log_dir: str) -> Path:
    address = getattr(worker, "address", None)
    name = getattr(worker, "name", None)
    identifier = address if address is not None else name
    return Path(log_dir) / f"{_safe_log_name(identifier)}.log"


def _redirect_output(log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1)
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file


def dask_setup(worker):
    log_dir = os.environ.get("COUPLING_SOURCES_WORKER_LOG_DIR")
    if log_dir is None:
        log_path = os.environ["COUPLING_SOURCES_RUN_LOG"]
    else:
        log_path = str(_worker_log_path(worker, log_dir))

    worker._coupling_sources_log_file = _redirect_output(log_path)
    print(f"worker log start: {getattr(worker, 'address', 'unknown')}", file=sys.stderr)


def dask_teardown(worker):
    log_file = getattr(worker, "_coupling_sources_log_file", None)
    if log_file is not None:
        log_file.flush()
