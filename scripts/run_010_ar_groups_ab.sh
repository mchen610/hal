#!/usr/bin/env bash
set -euo pipefail

# Controlled Fox/Fox behavior-cloning A/B. Both runs see the same windows,
# optimizer settings, and token budget; only the controller output head differs.
common_args=(
  --cfg.L-ctx 300
  --cfg.n-heads 4
  --cfg.batch-size 512
  --cfg.lr 0.001
  --cfg.max-steps 4096
  --cfg.warmup-steps 200
  --cfg.val-every 512
  --cfg.ckpt-every 1024
  --cfg.eval-every 0
  --cfg.eval-max-frames 3600
  --cfg.eval-replicas 8
  --cfg.eval-max-parallel 8
  --cfg.character-pair 1 1
  --cfg.windows-per-replay 64
)

uv run experiments/009_simplify_arch.py \
  "${common_args[@]}" \
  --comment fox-vs-fox-independent-ab

uv run experiments/010_ar_groups.py \
  "${common_args[@]}" \
  --comment fox-vs-fox-ar-groups-ab
