from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpyro

from coupling_sources.core.data import (
    GainConstants,
    MCMCSpecs,
    StimulusSpec,
    prepare_simulation_for_fit,
)
from coupling_sources.core.inference import fit_model, simulate_for_inference
from coupling_sources.core.stimuli import generate_stimulus
from coupling_sources.utils.artifacts import Artifacts

ARTIFACT_NAMES = (
    "fit_data.npz",
    "posterior.npz",
    "job.log",
)
CHILD_TERMINATION_TIMEOUT_SECONDS = 5.0
_CHILD_PROCESSES: dict[tuple[str, str], asyncio.subprocess.Process] = {}


@dataclass(frozen=True)
class JobResult:
    status: dict[str, Any]
    artifacts: dict[str, bytes]


async def run_job(
    job_id: str,
    job_config: Mapping[str, Any],
    mcmc: MCMCSpecs,
    progress_bar: bool,
    env: Mapping[str, str],
    run_token: str,
) -> JobResult:
    """Run one fit in a clean interpreter and return its compressed artifacts."""
    with tempfile.TemporaryDirectory(prefix=f"coupling-sources-{job_id}-") as temp:
        work_dir = Path(temp)
        input_path = work_dir / "input.json"
        output_dir = work_dir / "output"
        input_path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "job_config": dict(job_config),
                    "mcmc": asdict(mcmc),
                    "progress_bar": progress_bar,
                }
            )
        )

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "coupling_sources.compute.worker",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            env={**os.environ, **env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        process_key = (run_token, job_id)
        _CHILD_PROCESSES[process_key] = process
        try:
            try:
                stdout, stderr = await process.communicate()
            except asyncio.CancelledError:
                await _terminate_child(process)
                raise
        finally:
            _CHILD_PROCESSES.pop(process_key, None)

        if process.returncode == 0 and (output_dir / "status.json").exists():
            status = json.loads((output_dir / "status.json").read_text())
        else:
            details = "\n".join(
                text.strip()
                for text in (_decode_output(stdout), _decode_output(stderr))
                if text.strip()
            )
            status = {
                "ok": False,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "error": (
                    f"fit child exited with code {process.returncode}"
                    + (f"\n{details}" if details else "")
                ),
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            if details:
                with (output_dir / "job.log").open("a") as log_file:
                    log_file.write(details + "\n")

        return JobResult(
            status=status,
            artifacts={
                name: path.read_bytes()
                for name in ARTIFACT_NAMES
                if (path := output_dir / name).exists()
            },
        )


async def cancel_run(run_token: str) -> int:
    processes = [
        process
        for (process_run_token, _), process in _CHILD_PROCESSES.items()
        if process_run_token == run_token
    ]
    if processes:
        await asyncio.gather(*(_terminate_child(process) for process in processes))
    return len(processes)


async def _terminate_child(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return

    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(
            process.wait(),
            timeout=CHILD_TERMINATION_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()


def _decode_output(output: bytes | None) -> str:
    return "" if output is None else output.decode(errors="replace")


def _run_one_job(
    output_dir: Path,
    job_id: str,
    job_config: Mapping[str, Any],
    mcmc: MCMCSpecs,
    progress_bar: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "job.log"
    started_at = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", buffering=1) as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            print(f"fit start: {job_id}", flush=True)
            try:
                numpyro.enable_x64(True)
                stimuli = generate_stimulus(
                    StimulusSpec.from_mapping(job_config["stimulus"])
                )

                gains = GainConstants.from_mapping(job_config["gain_constants"])
                simulation_config = job_config["simulation"]
                simulation = simulate_for_inference(
                    stimuli=stimuli,
                    simulation_parameters=simulation_config["parameters"],
                    simulation_seed=int(simulation_config["seed"]),
                    gains=gains,
                )
                fit_data = prepare_simulation_for_fit(simulation)
                Artifacts.write_config(
                    output_dir / "fit_data.npz",
                    fit_data.to_config(),
                )

                fit_result = fit_model(
                    fit_data=fit_data,
                    c0=simulation_config["parameters"]["c0"],
                    gains=gains,
                    mcmc=mcmc,
                    progress_bar=progress_bar,
                    progress_label=job_id,
                )
                Artifacts.write_dist_samples(
                    output_dir / "posterior.npz",
                    fit_result["samples"],
                )
                Artifacts.write_mcmc_diagnostics(
                    output_dir / "posterior.npz",
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one posterior fit.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.input.read_text())
    status = _run_one_job(
        output_dir=args.output_dir,
        job_id=str(payload["job_id"]),
        job_config=payload["job_config"],
        mcmc=MCMCSpecs.from_mapping(payload["mcmc"]),
        progress_bar=bool(payload["progress_bar"]),
    )
    (args.output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
