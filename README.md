# coupling_sources

## Environment Creation

```bash
uv sync
source .venv/bin/activate
```

## Running Parameter Recovery

```bash
.venv/bin/python scripts/run_parameter_recovery.py \
  --config configs/parameters_sweep/adam.json \
  --compute-config configs/ssh_machines.json
```

The run writes artifacts under `artifacts/<config_name>_<YYYY_MM_DD_HH_MM_SS>/`: `config.json`, `logs.log`, shared `prior.zarr`, `job_grid.csv`, and `jobs/job_000000/{stimuli,simulation,posterior}.zarr`.

Remote SSH workers can be configured with per-host clone paths in `configs/compute/ssh_machines.example.json`. The runner starts the Dask scheduler on an SSH machine and can mix SSH workers with `local` workers in the same config. Each remote machine needs the repo cloned, dependencies installed, and the configured relative Python path available under its `project_path`; `host` may be either `hostname` or `user@hostname`. If the compute config includes any non-local host, the runner prompts once for the shared SSH password at startup.

Open the Dask dashboard while a remote run is active at `http://<scheduler-host>:8787/status`. The scheduler host is the first non-local machine in the compute config unless `scheduler_host` or `scheduler` is configured. The dashboard port is set by `scheduler_options.dashboard_address`. If that port is not reachable directly, run `ssh -L 8787:<scheduler-host>:8787 <user>@<scheduler-host>` and open `http://localhost:8787/status`.
