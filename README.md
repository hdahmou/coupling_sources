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
  --compute-config configs/compute/machines.json
```

The run writes artifacts under `artifacts/<config_name>_<YYYY_MM_DD>/`: `config.json`, `logs.log`, shared `prior.zarr`, `job_grid.csv`, and `jobs/job_000000/{stimuli,simulation,posterior}.zarr`.
