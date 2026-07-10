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
