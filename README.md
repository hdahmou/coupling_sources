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

Remote SSH workers can be configured with per-host clone paths in `configs/compute/ssh_machines.example.json`. The runner starts Dask scheduler/workers over SSH; each remote machine needs the repo cloned, dependencies installed, and the configured relative Python path available under its `project_path`. If the compute config includes any non-local host, the runner prompts once for the shared SSH password at startup.
