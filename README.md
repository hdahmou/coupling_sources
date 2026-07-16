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

Each machine used for SSH execution must already have a clone of this repository and a working Python environment at the configured `python` path.
The simplest way to update code on remote machines is to run `git pull` inside each clone. If a machine cannot pull from git directly, copy only the tracked project files from your local clone with:

```bash
git ls-files -z | rsync -az --from0 --files-from=- ./ hdahmou@cpu-03-01.isc.cnrs.fr:/home/hdahmou/Projects/coupling_sources/
```

Change the SSH target and destination path to match the host's `project_path`. This does not copy ignored artifacts, local configs, virtual environments, or `.git/`.


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
.venv/bin/python scripts/run_parameter_recovery.py \
  --config configs/parameters_sweep/adam.json \
  --compute-config configs/compute/ssh_machines.json
```

If the compute config contains active remote machines, the runner asks once for the shared SSH password. Jobs are independent: the local machine prepares JSON job specifications, sends assigned inputs to each remote host over SSH, runs local process pools on those hosts, and fetches result archives when each host finishes.

Set `mcmc.parallel_chains_per_fit` in the run config to the number of MCMC chains that can run in parallel inside one fit. The runner applies it to worker environments as `XLA_FLAGS=--xla_force_host_platform_device_count=<N>`, which NumPyro/JAX needs when `chain_method` is `parallel`. This is separate from `max_concurrent_fits`, which controls how many independent fits run at the same time on a host.

Workers use a persistent JAX compilation cache at `.jax_cache` in each project clone. Identical model and input shapes can reuse compiled executables across worker processes and later runs on the same host. Set `JAX_COMPILATION_CACHE_DIR` in the global or per-machine `env` configuration to override the location.

Each fit runs in a fresh spawned process. The process generates and saves `stimuli.npz`, simulates and saves `simulation.npz`, runs the posterior fit, saves `posterior.npz`, and then exits. Exiting after one job releases all JAX and LLVM mappings before another job starts while `max_concurrent_fits` still limits the number of active fits.

Artifacts are written under `artifacts/<config_name>_<YYYY_MM_DD_HH_MM_SS>/`:

- `config.json`, `logs.log`, `prior.npz`, and `job_grid.csv`
- `jobs/job_000000/job.json`, `stimuli.npz`, `simulation.npz`, `posterior.npz`, `status.json`, and `job.log`
- temporary transfer archives under `transfer/`

No Dask scheduler, dashboard, worker ports, nanny ports, or SSH tunnel script are used.

## SSH Compute Config

Use `configs/compute/ssh_machines.json` for your private machine list. An example is `configs/compute/ssh_machines.example.json`.

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
