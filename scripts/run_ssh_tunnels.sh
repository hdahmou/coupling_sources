#!/usr/bin/env bash
set -euo pipefail

config_path="configs/ssh_machines.json"
dry_run=0

usage() {
  cat <<'USAGE'
Usage: scripts/run_ssh_tunnels.sh [--config PATH] [--dry-run]

Starts the SSH tunnels needed when Dask workers cannot route directly to the
scheduler. The default compute config runs the scheduler on this machine and
uses a fixed localhost port:

  "scheduler_host": "local",
  "scheduler_options": {
    "host": "127.0.0.1",
    "port": 8786,
    "contact_address": "tcp://127.0.0.1:8786",
    "dashboard_address": ":8787"
  }

The script opens reverse forwards from each SSH worker back to this machine.
For non-local scheduler_host values, it also opens local forwards to the remote
scheduler host.

Keep the script running while the Dask job is active. Press Ctrl-C to stop the
tunnels.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

while (($#)); do
  case "$1" in
    -c|--config)
      shift
      (($#)) || die "--config requires a path"
      config_path="$1"
      ;;
    --dry-run)
      dry_run=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
  shift
done

[[ -f "$config_path" ]] || die "config file not found: $config_path"
require_command jq
require_command ssh
jq empty "$config_path" >/dev/null

scheduler_port="$(jq -r '.scheduler_options.port // empty' "$config_path")"
[[ "$scheduler_port" =~ ^[0-9]+$ ]] || die "set scheduler_options.port to a fixed port in $config_path"

scheduler_bind_host="$(jq -r '.scheduler_options.host // empty' "$config_path")"
case "$scheduler_bind_host" in
  0.0.0.0|127.0.0.1|localhost) ;;
  *) die "set scheduler_options.host to 0.0.0.0 or 127.0.0.1 in $config_path" ;;
esac

expected_contact_address="tcp://127.0.0.1:${scheduler_port}"
contact_address="$(jq -r '.scheduler_options.contact_address // empty' "$config_path")"
[[ "$contact_address" == "$expected_contact_address" ]] || die "set scheduler_options.contact_address to ${expected_contact_address} in $config_path"

dashboard_address="$(jq -r '.scheduler_options.dashboard_address // ":8787"' "$config_path")"
dashboard_port="${dashboard_address##*:}"
[[ "$dashboard_port" =~ ^[0-9]+$ ]] || die "could not parse dashboard port from scheduler_options.dashboard_address=${dashboard_address}"

scheduler_target="$(
  jq -r '
    def bare: tostring | split("@") | last;
    def remote_machine:
      (.host | bare) as $host
      | (["local", "localhost", "127.0.0.1", "::1"] | index($host) | not);
    def target:
      .host as $host
      | (.username // .user // (if ($host | tostring | contains("@")) then ($host | split("@")[0]) else "" end)) as $user
      | ($host | bare) as $bare
      | if (($user | length) > 0 and (($host | tostring | contains("@")) | not))
        then "\($user)@\($bare)"
        else $host
        end;
    def machine_for($host): [.machines[] | select((.host | bare) == ($host | bare)) | target][0] // empty;
    if .scheduler.host? then
      .scheduler as $scheduler
      | $scheduler.host as $host
      | ($scheduler.username // $scheduler.user // (if ($host | tostring | contains("@")) then ($host | split("@")[0]) else "" end)) as $user
      | ($host | bare) as $bare
      | if (($user | length) > 0 and (($host | tostring | contains("@")) | not))
        then "\($user)@\($bare)"
        else (machine_for($host) // $host)
        end
    elif .scheduler_host? then
      machine_for(.scheduler_host) // .scheduler_host
    else
      [.machines[] | select(remote_machine) | target][0] // empty
    end
  ' "$config_path"
)"
[[ -n "$scheduler_target" ]] || die "could not determine scheduler host from $config_path"
scheduler_bare="${scheduler_target##*@}"
scheduler_is_local=0
case "$scheduler_bare" in
  local|localhost|127.0.0.1|::1) scheduler_is_local=1 ;;
esac
if ((scheduler_is_local)); then
  case "$scheduler_bind_host" in
    127.0.0.1|localhost) ;;
    *) die "set scheduler_options.host to 127.0.0.1 when scheduler_host is local" ;;
  esac
fi

mapfile -t raw_worker_targets < <(
  jq -r --arg scheduler_bare "$scheduler_bare" '
    def bare: tostring | split("@") | last;
    def remote_machine:
      (.host | bare) as $host
      | (["local", "localhost", "127.0.0.1", "::1"] | index($host) | not);
    def target:
      .host as $host
      | (.username // .user // (if ($host | tostring | contains("@")) then ($host | split("@")[0]) else "" end)) as $user
      | ($host | bare) as $bare
      | if (($user | length) > 0 and (($host | tostring | contains("@")) | not))
        then "\($user)@\($bare)"
        else $host
        end;
    .machines[]
    | select(remote_machine)
    | select((.host | bare) != $scheduler_bare)
    | target
  ' "$config_path"
)

worker_targets=()
declare -A seen_worker_targets=()
for target in "${raw_worker_targets[@]}"; do
  [[ -n "$target" ]] || continue
  [[ -z "${seen_worker_targets[$target]:-}" ]] || continue
  seen_worker_targets[$target]=1
  worker_targets+=("$target")
done

safe_name() {
  local value="$1"
  value="${value//[!A-Za-z0-9_.-]/_}"
  printf '%s' "${value:-tunnel}"
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

control_dir=""
tunnel_targets=()
tunnel_sockets=()
tunnel_labels=()

cleanup() {
  trap - EXIT INT TERM
  if ((${#tunnel_targets[@]})); then
    echo "Stopping SSH tunnels..."
    for index in "${!tunnel_targets[@]}"; do
      echo "Stopping ${tunnel_labels[$index]}: ${tunnel_targets[$index]}"
      ssh -S "${tunnel_sockets[$index]}" -O exit "${tunnel_targets[$index]}" >/dev/null 2>&1 || true
    done
  fi
  [[ -z "$control_dir" ]] || rm -rf "$control_dir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_tunnel() {
  local label="$1"
  local target="$2"
  shift 2
  local socket="${control_dir}/$(safe_name "$label").sock"
  local command=(
    ssh -MNf
    -S "$socket"
    -o ControlMaster=yes
    -o ControlPersist=no
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    "$@"
    "$target"
  )

  echo "Starting ${label}: ${target}"
  if ((dry_run)); then
    print_command "${command[@]}"
    return
  fi

  "${command[@]}"
  tunnel_targets+=("$target")
  tunnel_sockets+=("$socket")
  tunnel_labels+=("$label")
  ssh -S "$socket" -O check "$target" >/dev/null 2>&1 || die "SSH tunnel failed health check: ${label}"
}

if ((dry_run)); then
  control_dir="/tmp/coupling_sources_tunnels.DRY_RUN"
else
  control_dir="$(mktemp -d "${TMPDIR:-/tmp}/coupling_sources_tunnels.XXXXXX")"
fi

echo "Scheduler host: ${scheduler_target}"
echo "Scheduler port: ${scheduler_port}"
echo "Dashboard URL: http://127.0.0.1:${dashboard_port}/status"
if ((${#worker_targets[@]})); then
  echo "Reverse tunnel workers: ${worker_targets[*]}"
else
  echo "Reverse tunnel workers: none"
fi

if ((scheduler_is_local)); then
  echo "Scheduler tunnel: none (scheduler runs on this machine)"
else
  scheduler_forwards=(
    -L "127.0.0.1:${scheduler_port}:127.0.0.1:${scheduler_port}"
  )
  if [[ "$dashboard_port" != "$scheduler_port" ]]; then
    scheduler_forwards+=(
      -L "127.0.0.1:${dashboard_port}:127.0.0.1:${dashboard_port}"
    )
  fi
  start_tunnel "scheduler_${scheduler_bare}" "$scheduler_target" "${scheduler_forwards[@]}"
fi

for worker_target in "${worker_targets[@]}"; do
  start_tunnel "worker_${worker_target##*@}" "$worker_target" \
    -R "127.0.0.1:${scheduler_port}:127.0.0.1:${scheduler_port}"
done

if ((dry_run)); then
  echo "Dry run only; no tunnels started."
  exit 0
fi

echo "SSH tunnels are running. Keep this process open while the Dask job runs."
echo "Press Ctrl-C to stop the tunnels."
while true; do
  sleep 3600 &
  wait $!
done
