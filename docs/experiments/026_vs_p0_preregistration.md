# 026 versus P0: pre-registration

Status: not launched

Written: 2026-08-16

Write the results section only after the run finishes. Do not edit the sections above it.

## Question

Does the 026 temporal-MTP package beat the P0 baseline in closed-loop play?

This is a **package** comparison, not a test of temporal factorization alone. See Confounds.

## Hypothesis

P0 fails at multi-frame commitments (recovery, edgeguard) while matching human per-frame
statistics. 026 models a joint future controller sequence and executes four frames per replan,
so it can represent a committed plan. If that is the binding constraint, 026 converts damage
into stocks at a higher rate than P0.

## Evidence that motivates it

Measured on the P0 Fox-ditto checkpoint (`notebooks/lesson_02_dumb_baseline.py`,
`notebooks/lesson_03_independence.py`, W&B `0xsfslbk`):

- P0 beats a previous-controller Markov baseline by 0.82 bits per frame, and earns all of it on
  frames where the action changes (2.5 versus 5.4 bits) rather than on held frames
  (0.088 versus 0.116).
- Sampled action-change rate is 1.05 times the human rate. The policy is not too passive.
- Entropy minus NLL is under 0.01 bits per group. The policy is not overconfident.
- Independent group sampling produces a controller absent from human Fox/Fox play on 0.34% of
  frames. Within-frame independence is a small effect.
- Closed loop: it needs 158 damage to take a stock and loses one every 78 damage taken, while
  net damage per minute is not distinguishable from zero (LCB -10.262).

Per-frame behavior is good and multi-frame outcomes are bad. That is what 026 targets.

## Single changed variable

The trained experiment file: `experiments/023_mtp_heads.py` to `experiments/026_temporal_mtp.py`.

Everything the launcher, data, seed, and evaluation protocol control stays fixed.

## Confounds, stated before the run

026's defaults differ from P0 on five axes at once. A win does **not** isolate temporal
factorization.

| Axis | P0 (023) | 026 |
| --- | --- | --- |
| `d_model` | 256 | 384 |
| `L_ctx` | 256 | 128 |
| `head_offsets` | 4 offsets, max 13 | 10 offsets, max 20 |
| `exec_horizon` | 1 | 4 |
| `batch_size` | 512 | 1024 |

Compute is **not** confounded: both process 131,072 frames per step (512 x 256 = 1024 x 128) and
both run 16,384 steps.

Record exact parameter counts for both. If 026 carries more than 15% more parameters, a matched
capacity control is required before any architecture claim.

## Hold fixed

1. Data: `data/processed/ranked-anonymized-1/mds-policy-v7`, full corpus, no character filter.
2. `seed=0`, `eval_seed=0`, `max_steps=16384`.
3. Closed-loop protocol: 96 final matchups, `eval_max_frames=7200`, 32 concurrent boots.
4. Decode temperature 1.0, no button support masking, no min-p.
5. The P0 reference checkpoint below. Do not retrain P0.

### Why not Fox/Fox only

Use the full corpus because the P0 reference trained on it, so the H2H is fair.

Do **not** use the earlier "387 passes will overfit" argument. It is wrong. The loader re-cuts
different random windows from each replay on every pass, so a pass is not a repeat of the same
examples. The Fox-ditto run's validation curve fell 1.404, 1.211, 1.155, 1.133 at steps 256, 512,
768, and 1,024 and was still dropping steeply when `max_steps` ended it. The Fox-only data limit
has never been reached and is unmeasured.

Whether a Fox-only policy beats a full-corpus policy is a separate open question. See
[Fox-only data limit](fox_only_data_limit.md).

## Reference checkpoint

P0 is W&B run `obx3o3az`. Final checkpoint SHA-256
`5d12d010fa3acd1ec07bd86a8e85d2cbb84c584a77b9b79e90dc6fcf03c32e4b`. Its official 96-boot
evaluation reported 0.777 stocks taken and 1.468 stocks lost per active minute.

Verify the SHA-256 before the H2H sweep. A mismatch aborts the comparison.

## Primary metric and decision threshold

**Primary:** paired mean stock difference over 64 mirrored H2H configurations, 026 minus P0, from
`hal/scripts/h2h.py`.

Mirrored H2H is the primary instrument because it removes the CPU-9 opponent, the stage draw, and
the port assignment as sources of variance. Prior rejections used it: P1 scored -0.062 and E1
scored -0.391, and neither was promoted.

| Outcome | Rule |
| --- | --- |
| Promote 026 | paired mean stock difference > +0.25 **and** its 95% CI lower bound > 0 |
| Inconclusive | CI lower bound <= 0 <= CI upper bound |
| Reject 026 | CI upper bound < 0 |

**Secondary, reported but not decisive:** net stocks per active minute against CPU-9 with its
cluster bootstrap LCB; damage per stock taken and damage per stock lost.

P0 Fox-ditto needed 158 damage per stock taken and lost one per 78 damage. A promotion should move
those toward each other. If the paired H2H promotes but this ratio does not improve, say so in the
results and do not claim the mechanism.

**Not decisive under any circumstance:** validation action NLL. P0 already reaches 1.133 bits while
losing. E2-S reached 1.134 at step 2,048. A better NLL alone does not promote.

## What would falsify the hypothesis

026 trains to completion, reaches equal or better validation NLL, and its paired mean stock
difference against P0 has a 95% CI upper bound below zero. That result says a joint temporal model
with a four-frame execution horizon does not fix the stock deficit, and the next hypothesis must be
compounding error rather than plan representation.

## Free within-checkpoint ablation

`experiments/026_temporal_mtp.py` accepts `exec_horizon` in `(4, 6)` only; `validate_config`
rejects 1. Evaluate the same finished checkpoint at both 4 and 6 and report both.

This does not isolate "plan versus no plan", because horizon 1 is unavailable. It measures only
how sensitive the policy is to replanning frequency. State that limit when reporting it.

## Commands

Train:

```bash
uv run scripts/launch_modal.py -- uv run experiments/026_temporal_mtp.py
```

Head-to-head, after the run uploads its final checkpoint:

```bash
uv run hal/scripts/h2h.py \
  --model-a.name 026-temporal \
  --model-a.checkpoint runs/<026-run>/final.pt \
  --model-a.experiment experiments.026_temporal_mtp \
  --model-b.name p0 \
  --model-b.checkpoint runs/<p0-run>/final.pt \
  --model-b.experiment experiments.023_mtp_heads \
  --n-configs 64
```

## Budget

E2 targeted 3.0 to 3.5 hours per arm on one RTX 4090 at about $0.78 per hour. 026 is wider but
carries half the context, so expect 3 to 5 hours and $3 to $6 for training plus evaluation. The
H2H sweep adds roughly 30 to 60 minutes.

Stop and re-plan if startup exceeds 30 minutes, if the warm step exceeds 0.5 seconds, or if the
projected total exceeds 6 hours.

## Post-run record

Fill in only after the run completes.

```text
Exact commit and command:
W&B run id:
Checkpoint path and SHA-256:
Parameter counts, 026 and P0:
Final validation NLL:
CPU-9: stocks taken/min, stocks lost/min, net with LCB:
Damage per stock taken, damage per stock lost:
Paired mean stock difference vs P0, 64 configs, with 95% CI:
exec_horizon 4 versus 6:
Decision:
Unexpected observations:
```
