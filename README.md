# coupling_sources

## Create the environment

Create the same environment from the repository root on the coordinator and
every Dask worker host:

```bash
uv sync --frozen
source .venv/bin/activate
```

Each host must use the same repository revision and `uv.lock` file.

## Run with Dask

Run these commands from the repository root with the environment activated.

On the scheduler host:

```bash
coupling-sources-scheduler --host 0.0.0.0
```

On every worker host, replacing `scheduler-host` with the scheduler hostname or
IP address:

```bash
coupling-sources-worker tcp://scheduler-host:8786 --nworkers 4 --nthreads 1
```

On the coordinator host, after setting `scheduler_address` in
`configs/compute/dask.json` to the same scheduler address:

```bash
coupling-sources-coordinator \
  --config configs/parameters_sweep/adam.json \
  --compute-config configs/compute/dask.json
```
