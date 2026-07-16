# coupling_sources

## Setup

```bash
uv sync
uv run pre-commit install
source .venv/bin/activate
```

The pre-commit hook must be installed once per clone. To run Black manually
across all tracked Python files:

```bash
uv run pre-commit run --all-files
```

Each Dask worker machine must already have a synchronized clone of this
repository and a working Python environment. The simplest way to update code on
remote machines is to run `git pull` inside each clone. If a machine cannot pull
from git directly, copy only the tracked project files from your local clone
with:

```bash
git ls-files -z | rsync -az --from0 --files-from=- ./ hdahmou@cpu-03-01.isc.cnrs.fr:/home/hdahmou/Projects/coupling_sources/
```

Change the SSH target and destination path to match the remote clone. This does
not copy ignored artifacts, local configs, virtual environments, or `.git/`.


## Run Parameter Recovery

Simulation parameters are `gamma_y` and `gamma_z`: `0` means no use, `1` means
ideal-observer use, and negative values reverse the effect. A useful recovery
grid is `{0, 0.5, 1}`. Gain calibration uses both resultant screens and the exact
truth/probe mixture.

For a locked design or an existing dataset, generate fixed constants directly
from its parameters:

```bash
.venv/bin/python scripts/compute_gain_constants.py \
  --sigma 0.55 \
  --lam 0.8 \
  --resultant-floor-full 0.6 \
  --resultant-floor-leave-one-out 0.6 \
  --n-trials-full 28 \
  --n-trials-partial 12 \
  --n-monte-carlo-replicates 2000 \
  --max-relative-mc-se 0.01
```

The command prints a `gain_constants` block plus the complete
`gain_calibration` provenance record. It always reports relative Monte Carlo
SEs and approximate 95% Monte Carlo intervals. The optional
`--max-relative-mc-se 0.01` requires both gains to have at most 1% relative
Monte Carlo SE and returns a nonzero status otherwise. A fixed run config
includes both output blocks; `gain_calibration.mode` is `fixed`, and recovery
verifies that the rounded constants match the recorded estimates. The mode and
calibration provenance are required; recovery no longer accepts untraceable
fixed constants or an implicit calibration mode.
The recorded fixed calibration design must also match every stimulus design in
the run; use automatic mode for a grid that varies any calibration field.

For a recovery grid whose stimulus designs require different benchmark units,
omit `gain_constants` and request coordinator-side calibration:

```json
"gain_calibration": {
  "mode": "auto",
  "seed": 20260711,
  "n_monte_carlo_replicates": 2000
}
```

Each unique combination of trial counts, noise, coupling, coupled sources, and
resultant floors is calibrated once before jobs are dispatched. All parameter
draws for that design reuse the result. The resolved constants appear in each
`job.json`, in `job_grid.csv`, and with their Monte Carlo provenance in
`gain_calibrations.json`.

```bash
coupling-sources-coordinator \
  --config configs/parameters_sweep/adam.json \
  --compute-config configs/compute/dask.json
```

The runner connects to an externally managed Dask scheduler and submits every
fit immediately as a separate future. Dask assigns those futures dynamically;
the application does not divide jobs into host or submission chunks.

Set `mcmc.parallel_chains_per_fit` in the run config to the number of MCMC
chains that can run in parallel inside one fit. The runner applies it to each
fit subprocess as `JAX_NUM_CPU_DEVICES=<N>`, which NumPyro/JAX needs when
`chain_method` is `parallel`. Dask worker processes and threads independently
control how many fits run concurrently.

Workers use a persistent JAX compilation cache at `.jax_cache` relative to the
worker's launch directory. Identical model and input shapes can reuse compiled
executables across fit subprocesses and later runs on the same host. Set
`JAX_COMPILATION_CACHE_DIR` in the compute config's optional `env` object to
override the location.

Each Dask task starts one fresh Python subprocess. The subprocess generates the
stimuli, simulates responses, fits the posterior, compresses the artifacts, and
then exits. Exiting after one fit releases all JAX and LLVM mappings. The Dask
task returns the compressed artifacts to the coordinator, so workers do not
need access to a shared artifact filesystem.

Artifacts are written under `artifacts/<config_name>_<YYYY_MM_DD_HH_MM_SS>/`:

- `config.json`, `logs.log`, `prior.npz`, and `job_grid.csv`
- `jobs/job_000000/job.json`, `stimuli.npz`, `simulation.npz`, `posterior.npz`, `status.json`, and `job.log`

## Dask Compute

Start a scheduler before running parameter recovery:

```bash
coupling-sources-scheduler --host 0.0.0.0
```

Start workers from the repository root on every compute host. One worker process
with one thread corresponds to one concurrent fit; increase `--nworkers` for
additional fit capacity:

```bash
coupling-sources-worker tcp://scheduler-host:8786 --nworkers 4 --nthreads 1
```

The scheduler and worker commands delegate to Dask's native CLIs and accept the
same options. The coordinator is the project command that prepares and submits
parameter-recovery fits.

Cluster provisioning, SSH access, ports, authentication, and TLS are external
to this application. Every worker must use a compatible checkout and Python
environment containing `coupling-sources` and its dependencies.

The compute config only identifies the scheduler and optional environment
variables inherited by fit subprocesses:

```json
{
  "scheduler_address": "tcp://127.0.0.1:8786",
  "env": {
    "JAX_COMPILATION_CACHE_DIR": ".jax_cache"
  }
}
```
