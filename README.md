# coupling_sources

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Run Parameter Recovery On SSH Workers

Use two terminals.

Terminal 1 keeps the SSH tunnels open:

```bash
scripts/run_ssh_tunnels.sh --config configs/ssh_machines.json
```

Leave this running while the job runs. Press Ctrl-C here to close all SSH forwards started by the script.

Terminal 2 starts parameter recovery:

```bash
.venv/bin/python scripts/run_parameter_recovery.py \
  --config configs/parameters_sweep/adam.json \
  --compute-config configs/ssh_machines.json
```

The runner asks once for the shared SSH password when the compute config contains SSH machines.

Open the Dask dashboard during the run at:

```text
http://127.0.0.1:8787/status
```

Artifacts are written under `artifacts/<config_name>_<YYYY_MM_DD_HH_MM_SS>/`: `config.json`, `logs.log`, `prior.zarr`, `job_grid.csv`, and `jobs/job_000000/{stimuli,simulation,posterior}.zarr`.

Remote worker stdout/stderr is written on each worker machine under `<project_path>/artifacts/<run_name>/worker_logs/<host>/`. Those logs include worker-side prints and NumPyro progress bars.

## SSH Compute Config

The default SSH setup runs the Dask scheduler on the current machine and starts only workers over SSH:

```json
{
  "scheduler_host": "local",
  "scheduler_options": {
    "host": "127.0.0.1",
    "port": 8786,
    "contact_address": "tcp://127.0.0.1:8786",
    "dashboard_address": ":8787"
  }
}
```

Each remote machine entry needs a reachable `host`, its own `project_path`, and a usable Python path relative to that project path. The repo and environment must already exist on each remote machine.

Use `scripts/run_ssh_tunnels.sh --dry-run --config configs/ssh_machines.json` to inspect the SSH commands without starting tunnels. If SSH reports `remote port forwarding failed` or `address already in use`, choose another unused scheduler port and replace `8786` everywhere in the config.
