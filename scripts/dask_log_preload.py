from __future__ import annotations

import os
import sys
from pathlib import Path


def _redirect_output(log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1)
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file


def dask_setup(worker):
    log_path = os.environ["COUPLING_SOURCES_RUN_LOG"]
    worker._coupling_sources_log_file = _redirect_output(log_path)


def dask_teardown(worker):
    log_file = getattr(worker, "_coupling_sources_log_file", None)
    if log_file is not None:
        log_file.flush()
