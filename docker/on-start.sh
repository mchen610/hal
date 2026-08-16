#!/usr/bin/env bash
# vast.ai cloud lifecycle for a single training run. scripts/launch_vast.py reads
# this file and passes its contents inline as the instance's onstart command (NOT
# baked into the image — it runs before the repo is cloned, so baking would strand
# the box on a stale copy). The image ships only the environment (deps in
# /opt/venv), so this script lays the code in at the exact git SHA, runs one
# training command, and tears the box down — there is no persistent state on the
# instance: datasets are fetched from R2, checkpoints stream back to R2, and logs
# go to W&B during the run.
#
# Driven entirely by env injected at create time:
#   HAL_GIT_SHA          commit to check out
#   HAL_GIT_REMOTE       repo URL containing HAL_GIT_SHA
#   HAL_TRAIN_CMD_B64    base64 of the training command (base64 survives the env string)
#   AWS_*, WANDB_API_KEY  R2 + W&B credentials
#   GITHUB_TOKEN         optional; only set when the repo/image is private
#   HAL_FAILURE_ACTION   optional; "destroy" (default) or "stop" on boot/train failure
#   HAL_KEEP_ALIVE       optional; "1" disables all self-teardown (debug: leave box up)
#   HAL_BOOT_TIMEOUT_S   optional; kill bootstrap if clone/sync/fetch/preflight exceeds this
#   HAL_TRAIN_IDLE_TIMEOUT_S optional; kill training if train.log stops changing
# vast injects CONTAINER_ID + CONTAINER_API_KEY (a per-instance key) so the box
# can stop/destroy itself.
set -Eeuo pipefail

log() { echo "[on-start] $*"; }

# vast does NOT inject env into this on-start shell, so recover it from two sources:
#   - account env-vars (R2 + W&B secrets, set in the vast console) — written to
#     /etc/environment, deliberately kept out of the per-instance config so they
#     don't leak into `show instance`/extra_env.
#   - the per-run `-e` vars (HAL_GIT_SHA, HAL_TRAIN_CMD_B64, HAL_KEEP_ALIVE) — these
#     live in the container's PID 1 environ (NUL-separated).
# Export both so this script and the training child inherit them.
set -a
[ -r /etc/environment ] && . /etc/environment 2>/dev/null || true
set +a
if [ -r /proc/1/environ ]; then
  while IFS= read -r -d '' kv; do
    case "$kv" in AWS_*=* | WANDB_*=* | GITHUB_TOKEN=* | HAL_*=*) export "$kv" ;; esac
  done < /proc/1/environ
fi

# Vast removes container logs when a self-destroying instance exits. Mirror the
# complete lifecycle output to a file so the final success or failure path can
# preserve it in R2 before teardown.
lifecycle_log=/tmp/hal-lifecycle.log
: > "$lifecycle_log"
exec > >(tee -a "$lifecycle_log") 2>&1

log "env check: AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL:+set} WANDB_API_KEY=${WANDB_API_KEY:+set} HAL_GIT_SHA=${HAL_GIT_SHA:+set} HAL_GIT_REMOTE=${HAL_GIT_REMOTE:+set} HAL_TRAIN_CMD_B64=${HAL_TRAIN_CMD_B64:+set} HAL_FAILURE_ACTION=${HAL_FAILURE_ACTION:-destroy}"

upload_lifecycle_log() {
  outcome="$1"
  if [ -z "${AWS_ENDPOINT_URL:-}" ] || [ -z "${AWS_BUCKET:-}" ]; then
    log "WARN: cannot preserve lifecycle log: R2 endpoint or bucket is missing"
    return
  fi

  sha_tag="${HAL_GIT_SHA:-unknown}"
  sha_tag="${sha_tag:0:10}"
  instance_tag="${CONTAINER_ID:-unknown}"
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  key="runs/_launch_logs/${sha_tag}/${timestamp}_${instance_tag}_${outcome}.log"
  snapshot=$(mktemp)
  cp "$lifecycle_log" "$snapshot"
  if /opt/venv/bin/python - "$snapshot" "$key" <<'PY'
import os
import sys

import boto3

source, key = sys.argv[1:]
client = boto3.client(
    "s3",
    endpoint_url=os.environ["AWS_ENDPOINT_URL"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
client.upload_file(source, os.environ["AWS_BUCKET"], key)
PY
  then
    log "preserved lifecycle log: s3://${AWS_BUCKET}/${key}"
  else
    log "WARN: failed to preserve lifecycle log at s3://${AWS_BUCKET}/${key}"
  fi
  rm -f "$snapshot"
}

# Teardown, gated on HAL_KEEP_ALIVE so a debug run leaves the box SSH-able. $1 is the
# vast verb (stop|destroy), $2 a human reason for the log.
teardown() {
  if [ "${HAL_KEEP_ALIVE:-0}" = "1" ]; then
    log "HAL_KEEP_ALIVE=1 — leaving instance up ($2); destroy manually when done"
    return
  fi
  log "$2 — ${1}ing instance"
  # `destroy` prompts for confirmation; onstart has no TTY, so without -y it aborts and
  # leaves the box billing after a successful run. `stop` has no prompt (and no -y flag).
  yes_flag=""
  [ "$1" = destroy ] && yes_flag="-y"
  VAST_API_KEY="$CONTAINER_API_KEY" vastai "$1" instance "$CONTAINER_ID" $yes_flag || true
}

failure_action="${HAL_FAILURE_ACTION:-destroy}"
case "$failure_action" in
  destroy | stop) ;;
  *)
    log "WARN: invalid HAL_FAILURE_ACTION=${failure_action}; defaulting to destroy"
    failure_action="destroy"
    ;;
esac

# Any failure during boot (clone, sync, fetch) destroys the box by default so a stopped
# instance cannot keep billing disk. Use HAL_FAILURE_ACTION=stop or HAL_KEEP_ALIVE=1
# only when you intentionally want a failed box left around for debugging.
boot_failure() {
  code="$1"
  line="$2"
  trap - ERR TERM
  if [ -n "${boot_watchdog_pid:-}" ]; then
    kill "$boot_watchdog_pid" 2>/dev/null || true
    wait "$boot_watchdog_pid" 2>/dev/null || true
  fi
  log "boot failed (line ${line}, exit ${code})"
  upload_lifecycle_log boot-failed
  teardown "$failure_action" "boot failure"
  exit "$code"
}
trap 'boot_failure "$?" "$LINENO"' ERR
trap 'boot_failure 124 "$LINENO"' TERM

# Fail loud + early if the injected inputs are missing (e.g. env recovery found
# nothing) instead of dying obscurely mid-clone.
: "${HAL_GIT_SHA:?missing — vast -e env not recovered from /proc/1/environ}"
: "${HAL_TRAIN_CMD_B64:?missing — vast -e env not recovered from /proc/1/environ}"
: "${AWS_ENDPOINT_URL:?missing — R2 creds not recovered from /proc/1/environ}"

boot_timeout_s="${HAL_BOOT_TIMEOUT_S:-900}"
case "$boot_timeout_s" in
  '' | *[!0-9]*)
    log "WARN: invalid HAL_BOOT_TIMEOUT_S=${boot_timeout_s}; defaulting to 900"
    boot_timeout_s=900
    ;;
esac
if [ "$boot_timeout_s" -gt 0 ]; then
  boot_shell_pid=$$
  (
    sleep "$boot_timeout_s"
    log "bootstrap timeout after ${boot_timeout_s}s; terminating on-start"
    kill -TERM "$boot_shell_pid"
  ) &
  boot_watchdog_pid=$!
  log "bootstrap timeout: ${boot_timeout_s}s"
else
  boot_watchdog_pid=""
  log "bootstrap timeout disabled"
fi

# Persist creds so an interactive `ssh` peek (e.g. to --resume) sees them too.
# `|| true`: grep exits 1 on no match, which must not trip `set -e`.
env | grep -E '^(AWS_|WANDB_|GITHUB_TOKEN)=' >> /etc/environment || true

# Code at the exact SHA. Clone into a fresh /opt/hal (rm first so it's robust whether
# the image baked a repo there or not — the venv lives at /opt/venv, untouched). uv
# sync then installs the pure-Python hal into the prebuilt venv (fast, no compiler —
# a uv.lock mismatch would fail loud here and trip the trap).
log "cloning hal @ ${HAL_GIT_SHA}"
# Step out of /opt/hal before deleting it: the onstart shell can inherit /opt/hal as
# its cwd, and `rm -rf` on the cwd makes the next git command die with "Unable to read
# current working directory". Don't rely on the image WORKDIR being a safe value.
cd /
rm -rf /opt/hal
# Public repo clones anonymously; the ${GITHUB_TOKEN:+…@} prefix injects auth only
# if a token was set (private repo/image). Safe under `set -u`.
git_remote="${HAL_GIT_REMOTE:-https://github.com/ericyuegu/hal.git}"
case "$git_remote" in
  https://github.com/*) clone_url="https://${GITHUB_TOKEN:+${GITHUB_TOKEN}@}${git_remote#https://}" ;;
  *) clone_url="$git_remote" ;;
esac
log "clone remote: ${git_remote}"
git clone --quiet "$clone_url" /opt/hal
cd /opt/hal
git checkout --quiet "$HAL_GIT_SHA"
uv sync --locked

# Shared memory, provisioned BEFORE the (billed) fetch so a shm-incapable host aborts in
# seconds, not after downloading the ISO+dataset. vast's container /dev/shm defaults to 64MB.
# The trainer uses the 'file_system' tensor-sharing strategy, so per-batch worker IPC (several
# GB at high worker/batch counts) no longer rides /dev/shm — but StreamingDataset's own
# coordination arrays still need room beyond 64MB. Remount to 16g (compose uses the same), then
# FAIL FAST if it didn't take: a too-small /dev/shm otherwise only surfaces as a DataLoader
# worker dying at step 0 (RuntimeError: worker exited unexpectedly), wasting the whole boot.
mount -o remount,size=16g /dev/shm 2>/dev/null && log "/dev/shm -> 16g" || log "WARN: /dev/shm remount failed"
shm_mb=$(df -m /dev/shm | awk 'NR==2 {print $2}')
log "/dev/shm = ${shm_mb}MB"
if [ "${shm_mb:-0}" -lt 1024 ]; then
  log "FATAL: /dev/shm ${shm_mb}MB < 1GB (remount failed/undersized) — dataloader would die at step 0; aborting"
  boot_failure 1 "$LINENO"
fi

# Datasets/fixtures from R2 (sha-pinned, idempotent); stats.json sits outside the
# streamed shards so pull it up front. Shards stream lazily during training.
log "fetching fixtures + dataset stats"
uv run fetch
uv run python -c "from hal import streams; [streams.pull_stats(s) for s in streams.ALL]"

# Headless GL context for the closed-loop Dolphin eval.
pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &)
export DISPLAY=:99

# Preflight: the image ships CUDA 13. A host whose driver is too old (cuda_max_good < 13)
# fails torch's CUDA init (error 804: forward-compat unsupported on consumer GPUs), and the
# trainer's `DEVICE = "cuda" if is_available() else "cpu"` then silently runs ~100x slower on
# CPU and never self-stops. Assert here so a slipped-through box trips the boot trap -> stop.
log "preflight: asserting CUDA is available"
uv run python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available() is False — host driver too old for the CUDA-13 image'"

# DataLoader workers pass tensor-storage handles to the main process over a unix socket
# via file descriptors; num_workers * prefetch_factor in-flight batches can exceed the
# container's default open-fd soft limit (often 1024), crashing mid-run with
# "RuntimeError: received 0 items of ancdata" in recvfds. Raise the soft limit to the
# hard cap so fd-passing has headroom.
ulimit -n "$(ulimit -Hn)" && log "open files (ulimit -n) -> $(ulimit -n)"

if [ -n "$boot_watchdog_pid" ]; then
  kill "$boot_watchdog_pid" 2>/dev/null || true
  wait "$boot_watchdog_pid" 2>/dev/null || true
fi
trap - TERM

cmd="$(printf '%s' "$HAL_TRAIN_CMD_B64" | base64 -d)"
log "training: ${cmd}"
# Run training outside the trap so we can branch on its exit code. Success always
# destroys the box. Failure follows HAL_FAILURE_ACTION, which defaults to destroy
# because stopped instances still bill their disk.
set +e
idle_timeout_s="${HAL_TRAIN_IDLE_TIMEOUT_S:-900}"
case "$idle_timeout_s" in
  '' | *[!0-9]*)
    log "WARN: invalid HAL_TRAIN_IDLE_TIMEOUT_S=${idle_timeout_s}; defaulting to 900"
    idle_timeout_s=900
    ;;
esac
if [ "$idle_timeout_s" -gt 0 ]; then
  log "training idle timeout: ${idle_timeout_s}s without /opt/hal/train.log writes"
else
  log "training idle timeout disabled"
fi
: > /opt/hal/train.log
setsid bash -c "$cmd" > >(tee -a /opt/hal/train.log) 2>&1 &
train_pid=$!
code=0
while true; do
  if ! kill -0 "$train_pid" 2>/dev/null; then
    wait "$train_pid"
    code=$?
    break
  fi
  if [ "$idle_timeout_s" -gt 0 ]; then
    last_write=$(stat -c %Y /opt/hal/train.log 2>/dev/null || date +%s)
    now=$(date +%s)
    idle_s=$((now - last_write))
    if [ "$idle_s" -ge "$idle_timeout_s" ]; then
      log "training idle timeout: no /opt/hal/train.log writes for ${idle_s}s; terminating training process group"
      kill -TERM "-$train_pid" 2>/dev/null || kill -TERM "$train_pid" 2>/dev/null || true
      sleep 20
      kill -KILL "-$train_pid" 2>/dev/null || kill -KILL "$train_pid" 2>/dev/null || true
      wait "$train_pid" 2>/dev/null || true
      code=124
      break
    fi
  fi
  sleep 30
done
set -e

if [ "$code" -eq 0 ]; then
  upload_lifecycle_log training-succeeded
  teardown destroy "training succeeded (checkpoints in R2, logs in W&B)"
else
  upload_lifecycle_log training-exited-${code}
  teardown "$failure_action" "training exited ${code}"
fi
