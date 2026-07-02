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

The runner asks once for the shared SSH password when the compute config contains active SSH workers.

Open the Dask dashboard during the run at:

```text
http://127.0.0.1:8787/status
```

Artifacts are written under `artifacts/<config_name>_<YYYY_MM_DD_HH_MM_SS>/`: `config.json`, `logs.log`, `prior.zarr`, `job_grid.csv`, and `jobs/job_000000/{stimuli,simulation,posterior}.zarr`.

Remote worker stdout/stderr is written on each active worker machine under `<project_path>/artifacts/<run_name>/worker_logs/<host>/`. Task milestones and exceptions are also forwarded into the local `logs.log`; full raw worker output, including progress-bar writes, stays in the remote task log. The local `logs.log` prints the exact remote log directory before submitting tasks and the exact task log path when a fit finishes. Hosts with `max_concurrent_fits: 0` do not run tasks, so they do not create worker task logs.

Do not open the scheduler TCP address in a browser. `tcp://127.0.0.1:8786` is Dask's binary scheduler endpoint; the HTTP dashboard is `http://127.0.0.1:8787/status`.

## SSH Compute Config

The default SSH setup runs the Dask scheduler on the current machine and starts only workers over SSH. Because the remote machines cannot connect back directly, tunnel mode pins each remote worker/nanny to deterministic localhost ports and forwards those ports back to the launcher machine:

```json
{
  "scheduler_host": "local",
  "scheduler_options": {
    "host": "127.0.0.1",
    "port": 8786,
    "contact_address": "tcp://127.0.0.1:8786",
    "dashboard_address": ":8787"
  },
  "tunnel": {
    "enabled": true,
    "worker_port_start": 39000,
    "nanny_port_start": 40000
  }
}
```

Each remote machine entry needs a reachable `host`, its own `project_path`, and a usable Python path relative to that project path. The repo and environment must already exist on each remote machine. Set `max_concurrent_fits` to the number of workers to launch on that host; set it to `0` to keep the host in the config but disable it for the current run. Disabled hosts are skipped by Dask, but `scripts/run_ssh_tunnels.sh` still keeps scheduler reverse tunnels for all configured remote hosts.

In tunnel mode, worker slot `N` uses worker port `worker_port_start + N` and nanny port `nanny_port_start + N`. The backend starts one remote worker spec per slot so each worker advertises the matching forwarded localhost address.

Use `scripts/run_ssh_tunnels.sh --dry-run --config configs/ssh_machines.json` to inspect the SSH commands without starting tunnels. If SSH reports `remote port forwarding failed` or `address already in use`, choose unused scheduler and tunnel port ranges in the config.
