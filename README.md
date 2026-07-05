# coupling_sources

## Setup

```bash
uv sync
source .venv/bin/activate
```

Each machine used for SSH execution must already have a clone of this repository and a working Python environment at the configured `python` path.
The simplest way to update code on remote machines is to run `git pull` inside each clone. If a machine cannot pull from git directly, copy only the tracked project files from your local clone with:

```bash
git ls-files -z | rsync -az --from0 --files-from=- ./ hdahmou@cpu-03-01.isc.cnrs.fr:/home/hdahmou/Projects/coupling_sources/
```

Change the SSH target and destination path to match the host's `project_path`. This does not copy ignored artifacts, local configs, virtual environments, or `.git/`.

## Run Parameter Recovery

```bash
.venv/bin/python scripts/run_parameter_recovery.py \
  --config configs/parameters_sweep/adam.json \
  --compute-config configs/ssh_machines.json
```

If the compute config contains active remote machines, the runner asks once for the shared SSH password. Jobs are independent: the local machine prepares the artifact run, sends assigned job inputs to each remote host over SSH, runs local process pools on those hosts, and fetches result archives when each host finishes.

Artifacts are written under `artifacts/<config_name>_<YYYY_MM_DD_HH_MM_SS>/`:

- `config.json`, `logs.log`, `prior.zarr`, and `job_grid.csv`
- `jobs/job_000000/{stimuli,simulation,posterior}.zarr`
- `worker_logs/<host>/<job_id>.log`
- `job_status/<job_id>.json`
- temporary transfer archives under `transfer/`

No Dask scheduler, dashboard, worker ports, nanny ports, or SSH tunnel script are used.

## SSH Compute Config

Use `configs/ssh_machines.json` for your private machine list. an example is `ssh_machines.example.json`

Fields:

- `host`: SSH target, optionally including the username.
- `project_path`: absolute path to this repo clone on that machine.
- `python`: Python executable, relative to `project_path` unless absolute.
- `max_concurrent_fits`: number of independent posterior fits to run on that host; `0` keeps the host in the config but disables it.
- `env`: optional environment variables, either global or per machine.
- `connect_kwargs`: optional Fabric/Paramiko connection options. Do not put clear-text passwords here; the runner prompts at start.

To run only on the current machine, use a local machine entry:

```json
{
  "python": ".venv/bin/python",
  "machines": [
    {
      "host": "local",
      "max_concurrent_fits": 4
    }
  ]
}
```
