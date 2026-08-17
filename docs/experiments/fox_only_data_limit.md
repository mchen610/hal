# Fox-only: is 5,414 replays enough?

Status: not launched

Written: 2026-08-16

## Question

Does training P0 on Fox/Fox alone, to a real step budget, beat P0 trained on the full corpus when
both play Fox/Fox?

## Why it is open

The only Fox-only run, W&B `0xsfslbk`, used `max_steps=1024`. Its validation curve was:

| step | val action NLL (bits) |
| --- | --- |
| 256 | 1.404 |
| 512 | 1.211 |
| 768 | 1.155 |
| 1,024 | 1.133 |

It was still falling steeply when the step budget ended it. Nothing about the Fox-only data limit
has been measured. An earlier claim that 16,384 steps would overfit 5,414 replays is not supported:
the window sampler draws different windows from each replay on every pass, so a pass does not
repeat the same examples. The binding limit is the 5,414 unique games, and that limit is unknown.

## Hypothesis

A Fox-only policy spends no capacity on 25 other characters and sees matchup-specific timing on
every frame, so at equal steps it plays Fox/Fox better than the full-corpus P0.

The competing hypothesis is that most of what P0 learns (movement, ledge behavior, drift, hitstun
response) transfers across characters, so 21 times more data wins and specialization loses.

## Single changed variable

`--cfg.character-pair 1 1`. Everything else matches the P0 reference configuration.

## Hold fixed

1. `experiments/023_mtp_heads.py`, `seed=0`, `eval_seed=0`.
2. Data root `data/processed/ranked-anonymized-1/mds-policy-v7`.
3. Decode temperature 1.0, no support masking, no min-p.
4. Closed-loop protocol and matchup counts.
5. `--cfg.val-n-samples 48`. The Fox/Fox val split holds 52 replay rows, so a larger request
   exceeds the eligible set.

## Primary metric and decision threshold

**Primary:** paired mean stock difference over 64 mirrored H2H configurations against the full-corpus
P0 reference (`obx3o3az`, SHA-256
`5d12d010fa3acd1ec07bd86a8e85d2cbb84c584a77b9b79e90dc6fcf03c32e4b`).

Both policies play Fox in the sweep, so the comparison is direct.

| Outcome | Rule |
| --- | --- |
| Specialization wins | paired mean stock difference > +0.25 and 95% CI lower bound > 0 |
| Inconclusive | CI lower bound <= 0 <= CI upper bound |
| Specialization loses | CI upper bound < 0 |

**Secondary:** the validation curve itself. Record `val/loss` at every 512 steps. The step at which
it stops falling is the answer to "is 5,414 replays enough", independent of the H2H outcome.

**Not decisive:** a lower Fox-only validation NLL on its own. The Fox-only and full-corpus runs
validate on different populations, so their NLL values are not directly comparable. Only the H2H
compares them fairly.

## What would falsify the hypothesis

The validation curve flattens well before 8,192 steps **and** the paired H2H upper bound sits below
zero. That says the Fox/Fox corpus is too small to specialize on and cross-character data is doing
real work.

## Command

```bash
uv run scripts/launch_modal.py -- uv run experiments/023_mtp_heads.py \
  --cfg.character-pair 1 1 \
  --cfg.max-steps 8192 \
  --cfg.val-every 512 \
  --cfg.val-n-samples 48
```

Head-to-head after the checkpoint uploads:

```bash
uv run hal/scripts/h2h.py \
  --model-a.name fox-only \
  --model-a.checkpoint runs/<fox-only-run>/final.pt \
  --model-a.experiment experiments.023_mtp_heads \
  --model-b.name p0-full \
  --model-b.checkpoint runs/<p0-run>/final.pt \
  --model-b.experiment experiments.023_mtp_heads \
  --n-configs 64
```

## Budget

About 1.5 to 2 hours and $1.50 to $2 for training plus evaluation at 8,192 steps. The H2H sweep
adds roughly 30 to 60 minutes.

If `val/loss` is still falling at 8,192 steps, that is a result, not a failure. Record it and run a
longer arm before drawing a conclusion.

## Post-run record

```text
Exact commit and command:
W&B run id:
Checkpoint path and SHA-256:
val/loss at every 512 steps:
Step where val/loss stopped falling:
CPU-9: stocks taken/min, stocks lost/min, net with LCB:
Paired mean stock difference vs full P0, 64 configs, with 95% CI:
Decision:
Unexpected observations:
```
