# HAL research primer and working notebook

Last updated: 2026-08-16

This is the durable record of what we have learned while trying to train a
Melee policy. It is also a map for learning the codebase. It distinguishes
facts established by code or recorded experiments from our current
interpretations. When those disagree, trust the recorded evidence and run a
controlled experiment.

## Executive summary

1. The project is not currently bottlenecked by access to Fox/Fox data. The
   filtered training split contains 5,414 Fox/Fox replay rows and two player
   perspectives per replay.
2. Our models optimize action imitation, not winning. A low held-out action
   loss is necessary, but it has repeatedly failed to predict closed-loop
   strength.
3. The one-frame GPT family can learn locally plausible movement while still
   failing at recovery, survival, and coherent multi-frame decisions.
4. Experiment 023 is the official simple P0 baseline. Its four future heads
   are independent marginals and only offset 1 is executed. It is useful as a
   control, not the final architecture.
5. Experiment 026 is the important temporal successor: it models a joint
   future controller sequence and executes four-frame plans. Experiments 030,
   032, and 033 build on that line.
6. More epochs on experiment 023 are unlikely to solve its structural
   problem. Our Fox-only run already made about 24 passes through its sampled
   corpus.
7. RL is not a repair tool for an incoherent starting policy. Our PPO run
   found a narrow high-reward behavior (mostly up-smash) instead of learning
   generally good Melee.
8. The next clean scientific comparison is 023 versus 026 on the same Fox/Fox
   data, seeds, compute budget, and closed-loop evaluation.

## The complete system in one picture

```text
raw .slp replay
    |
    | peppi parses game-start, pre-frame, post-frame, and game-end events
    v
index.jsonl
    | one record per replay: players, characters, stage, path, outcome, etc.
    |
    | filter chooses eligible replay paths
    v
canonical MDS v7
    | one sample per replay; each field is an array over replay frames
    |
    | exact policy projection and packing
    v
compact mds-policy-v7
    | one compact row per replay
    |
    | dataloader selects ego, samples windows, relabels p1/p2 -> ego/opp,
    | normalizes features, and forms targets
    v
training batch
    | context frames -> policy -> future controller distribution
    |
    | behavior-cloning loss updates model parameters
    v
checkpoint (.pt)
    |
    | closed-loop policy receives live Dolphin frames and samples controllers
    v
Dolphin evaluation -> .slp replays + stock/damage metrics
```

The most important boundary is between offline training and closed-loop play.
Offline data always contains human states caused by human actions. During live
play, the model receives states caused by its own earlier sampled actions.

## Data sources and storage

### Public sources we found

- Raw Slippi replays: https://huggingface.co/datasets/erickfm/slippi-public-dataset-v3.7
- Processed frame dataset: https://huggingface.co/datasets/erickfm/frame-melee
- Small processed subset: https://huggingface.co/datasets/erickfm/frame-melee-subset
- The project README also links six large ranked-anonymized archives from the
  Slippi Discord community.

Those public datasets are useful, but the current HAL training path uses the
team's Cloudflare R2 bucket. R2 exposes an S3-compatible API, so the code and
CLI use AWS-style credentials even though the storage provider is Cloudflare.

### Why `aws s3 ls` was denied

`aws s3 ls` with no URI asks the account to list every bucket. The credentials
can be allowed to read a known bucket while being denied `ListBuckets`. That
is normal least-privilege behavior. A command against a known prefix can still
work:

```bash
aws s3 ls s3://hal/ --endpoint-url "$AWS_ENDPOINT_URL"
rclone lsf r2:hal/
```

`rclone` is convenient for browsing, recursive listings, copying one object,
and syncing directory trees. The R2 web UI is better for occasional manual
inspection. The CLI is better for reproducible scripts and large recursive
operations.

### What one "row" means

A raw `.slp` file is not tabular. It is a binary event stream containing:

- a game-start block;
- one pre-frame controller event per player and frame;
- one post-frame game-state event per player and frame;
- rollback corrections;
- a game-end block and metadata.

An MDS sample is one complete replay. Its fields look conceptually like this:

```text
schema_version                    scalar
frame                             int32[T]
stage                             int32[T]
p1_character                      int32[T]
p2_character                      int32[T]
p1_position_x                     float32[T]
p1_position_y                     float32[T]
p1_action                         int32[T]
p1_main_stick_x                   float32[T]
...
p2_...
```

`T` is the replay's frame count. Training does not treat the full replay as one
transformer example. The dataloader slices fixed-length windows from each row.

### Canonical MDS versus compact policy MDS

Canonical MDS v7 is the source of truth. It preserves a broad set of replay
fields. `mds-policy-v7` is an exact, packed projection containing only what the
policy training path needs.

Recorded upstream facts:

- 114,768 audited ranked replays;
- approximately 1.23 billion frames;
- 112,409 replay rows in the training split;
- 29.82 GB compressed for canonical v7;
- 13.34 GB compressed for compact policy v7;
- 802.47 GB versus 76.19 GB after decoding the relevant arrays;
- one 128-bit stable replay ID per replay;
- 512 distinct replay IDs per P0 batch;
- no replay returns in the immediately following batch;
- four sampled windows per replay;
- exact slice decoding rather than decoding a complete replay first.

See [data_pipeline.md](experiments/data_pipeline.md) and
[`hal/data/policy_schema.py`](../hal/data/policy_schema.py).

### Why train, validation, and test all exist

- Training data updates model parameters.
- Validation data selects configurations and detects overfitting while work is
  in progress.
- Test or final opponent sets should be touched only for final claims.

Ignoring the distinction allows accidental selection on the final result. For
quick exploratory work we sometimes do not care about a formal test claim, but
the code still preserves the split so serious comparisons remain possible.

### Current local data

At the time of this note, `data/` uses approximately 39 GB:

```text
14 MB  data/multishine_mds
35 MB  data/raw
1.3 GB data/emulator
38 GB  data/processed
```

Most raw/full duplicate data was removed earlier. The ranked 10k processed
dataset remains because it is still useful for smaller experiments. The
compact policy artifact streams missing shards from R2 when needed.

## IDs: the bug we found and the final convention

### There are different character ID spaces

Slippi's game-start block uses Melee's external/character-select IDs. Examples:

```text
Captain Falcon = 0
Donkey Kong    = 1
Fox            = 2
Marth          = 9
Falco          = 20
Dr. Mario      = 22
Ganondorf      = 25
```

Libmelee's `melee.Character` is the internal/in-game ID space reported in
post-frame game state. Examples include Fox=1 and Falco=22. Casting an external
ID directly to `melee.Character` is wrong.

The external order is the game's canonical character-select order. It is not
alphabetical, tier-based, or a HAL design decision. Reference implementations
include py-slippi's `CSSCharacter` enum:

https://github.com/hohav/py-slippi/blob/master/slippi/id.py

### The final HAL rule

HAL converts external character IDs to `melee.Character` at the peppi read
boundary. Everything downstream speaks the internal libmelee value:

- replay index;
- canonical MDS;
- compact policy MDS;
- filters;
- model features;
- live Dolphin observations.

The source of truth is [`hal/wire.py`](../hal/wire.py). The conversion is not
an identity map. Fox is the simplest witness: external 2 becomes internal 1.

Nana has no selectable external character ID. Ice Climbers external ID 14 maps
to Popo; Nana is represented as the follower in per-frame game state.

### Stage IDs also differ

Slippi stage IDs and `melee.Stage` values are not generally equal. Fountain of
Dreams is the canonical warning: Slippi 2 versus libmelee 8. Stage conversion
also happens at the input boundary through `slp_stage_to_libmelee`.

### Static character versus live character

MDS has two distinct concepts:

- `p1_character` / `p2_character`: replay-level game-start character metadata,
  broadcast over frames;
- `character_live`: the engine's per-frame active body.

`character_live` changes when Zelda and Sheik transform. The game-start value
is static metadata: regardless of how the initial Zelda/Sheik state is encoded,
it cannot represent transformations later in the match.

For Ice Climbers, Popo and Nana already have separate player/follower frame
blocks. For normal selectable characters, the practical transform ambiguity is
primarily Zelda/Sheik.

Schema v6 added `character_live`; schema v7 retained it and added rank. P0's
base observation bundle does not consume every v6 feature. A richer bundle
must opt into the live character field.

### Did the old bug ruin every model?

No single answer applies to every checkpoint:

- Current v5+ canonical data and v7 policy data normalize characters correctly.
- Older checkpoints trained against older MDS semantics keep whatever IDs were
  in that data. A code fix does not retroactively change their embeddings.
- If a model trained and evaluated with the same wrong IDs, it could still learn
  a consistent arbitrary embedding label. The larger risk was mismatch between
  offline data and live evaluation, or filters selecting the wrong matchup.
- Stage had already been converted; character had not. That asymmetry made the
  original code especially easy to misunderstand.

## What the policy observes

The exact feature bundle depends on the experiment. The common base frame token
contains normalized game state for:

- ego;
- ego's Nana follower, masked when absent;
- opponent's Nana follower, masked when absent;
- opponent;
- ego controller history;
- static ego/opponent character metadata;
- stage metadata.

Common game-state features include position, percent, shield, stocks, facing,
action state, hitlag, jumps, airborne state, and hurtbox state. Newer v6/v7
bundles can add engine velocity, state age, live character, state flags,
L-cancel state, ground/platform, item slots, and engineered spatial features.

An observation, often called `obs`, is simply the model-facing representation
of the current and recent game state. During live play, `Session.step` returns a
canonical frame dictionary. `RecedingHorizon` maintains a rolling context and
turns those frames into the same tensor layout used by training.

## What the policy outputs

The logical controller vector has 14 channels, defined in
[`hal/wire.py`](../hal/wire.py):

```text
main_stick_x, main_stick_y,
c_stick_x, c_stick_y,
trigger_l, trigger_r,
button_a, button_b, button_x, button_y,
button_z, button_r, button_l, button_d_up
```

Start is intentionally absent so the model cannot pause the game.

The categorical GPT experiments quantize this vector into four groups:

| Group | Classes | Meaning |
| --- | ---: | --- |
| buttons | 256 | Eight binary button bits as one combination |
| main stick | 65 | Clustered two-dimensional stick location |
| C-stick | 9 | Clustered two-dimensional C-stick location |
| triggers | 25 | Joint left/right analog trigger centers |

Experiment 009 and P0/023 sample those groups independently given the
transformer state. That can combine individually likely group choices into a
controller frame that was not jointly likely in human play. Later decoders
condition groups on prior groups to address this.

## Training terminology

### Behavior cloning (BC)

Behavior cloning means supervised imitation:

```text
input:  recent game states and controller history
target: the human's next controller action
loss:   negative log probability of that recorded action
```

BC answers, "What would a human probably press here?" It does not directly
answer, "Which action maximizes the chance of winning?"

### Context, target, window, row, and epoch

- Row: one replay stored in MDS.
- Window: a slice sampled from a replay row.
- Context: the past frames presented to the transformer.
- Target: one or more future controller frames.
- Batch: many windows used in one gradient estimate.
- Step: one optimizer update.
- Epoch/pass: enough sampled windows to cover roughly one corpus worth of rows.

The five-second clip experiment changed window sampling. At 60 frames/second,
`L_ctx=300` is approximately five seconds. It did not rewrite MDS so that each
five-second clip became an independent stored row. Shorter rows or more windows
change data mixing; they do not change a one-frame imitation objective into a
planning objective.

### Batch size 128 versus 512

Batch size is the number of windows contributing to one optimizer update.

- Larger batches give a less noisy estimate of the average gradient.
- Larger batches process more frames per step and usually need fewer steps for
  the same number of seen examples.
- Smaller batches provide noisier updates and more optimizer steps per corpus
  pass.
- Step counts cannot be compared without multiplying by batch size and context.

P0 uses batch 512 and context 256, so a dense step contains 131,072 frame
positions before accounting for padding. That is why 32,768 steps on a small
subset can be extreme overtraining even if "32k" sounds modest.

### Loss, NLL, and log probability

If the model gives the recorded action probability `p`, its negative log
likelihood is `-log(p)`:

- probability near 1 -> loss near 0;
- probability near 0 -> large loss.

Gradient descent changes parameters so recorded actions receive larger log
probabilities. In policy-gradient RL, sampled actions with positive advantage
have their log probabilities increased; sampled actions with negative
advantage have them decreased.

### Temperature

Temperature changes sampling sharpness:

- below 1: concentrate on high-probability actions;
- 1: sample the learned distribution directly;
- above 1: flatten the distribution and increase randomness;
- greedy argmax: always select the highest-logit action.

Greedy decode is not automatically stronger. In these policies it often falls
into a repeated do-nothing mode because the most common per-frame action is a
hold or neutral action.

## Model families we have touched

### Experiment 001: flow matching

[`experiments/001_flow_matching_baseline.py`](../experiments/001_flow_matching_baseline.py)
predicts a continuous action chunk by learning a velocity field that transforms
noise into controller sequences. It predicts 16 frames and integrates for
eight flow steps. It is a genuine flow-matching experiment, but it is an older
open-loop baseline, not proof that flow matching solved Melee.

### Experiment 009: GPT next-frame policy

[`experiments/009_simplify_arch.py`](../experiments/009_simplify_arch.py) is a
nanoGPT/GPT-2-style causal transformer:

- one token per frame;
- 256-frame context in the original run;
- hidden size 256, eight layers, four heads;
- predicts the next controller frame;
- replans every frame;
- independently samples the four controller groups.

Our original Fox/Fox checkpoint was:

```text
runs/260615-041800_gpt-d256-L8-h4-Lc256_ranked-anon-1_fox-vs-fox/final.pt
```

It looked subjectively better than several later models, but it lacks the
standardized evidence needed for a scientific comparison. It trained to step
32,768 on a small matchup subset, so overtraining is a serious concern.

### Five-second clip variant

The later clip run is:

```text
runs/260805-212129_009_simplify_arch_gpt-d256-L8-h4-Lc300_ranked-anon-1_gpt-clip5s-k64-b512-4k-fox-vs-fox/final.pt
```

It used about a five-second context and many windows per replay. Qualitative
play was poor. The result shows that more short clips alone do not repair
per-frame factorization, objective mismatch, or closed-loop compounding error.

### Experiment 023: P0 independent MTP heads

[`experiments/023_mtp_heads.py`](../experiments/023_mtp_heads.py) is upstream's
simple, carefully measured P0 baseline:

- 6.82 million parameters;
- context 256;
- hidden size 256, eight layers, four heads;
- full causal attention;
- independent output heads for offsets 1, 5, 9, and 13;
- loss is primary offset 1 plus the mean auxiliary loss;
- only offset 1 controls Dolphin;
- no critic, AWR, rank weighting, temporal action conditioning, or within-frame
  action conditioning.

The official full-data P0 trained for 16,384 steps. Its final offset-1 NLL was
1.029 bits/frame. Its official 96-boot evaluation produced:

```text
stocks taken/min: 0.777
stocks lost/min:  1.468
damage dealt/min: 129.6
damage taken/min: 116.4
terminal games won: 0 of 24
```

This is a valid baseline, but not a strong Melee agent.

### E1 and E2 controls around P0

E1 added a state-only MLP output adapter. It slightly improved offline NLL and
its unpaired CPU point estimate, but lost the paired H2H comparison against P0.
It was not promoted.

E2 added within-frame autoregressive group conditioning. Its stable-order run
was canceled around step 2,300. It has no final checkpoint or closed-loop
decision. It cannot be cited as a success or failure.

This is important research discipline: implemented code is not an experimental
result, and a partial checkpoint is not a final result.

### Experiment 026: temporal MTP

[`experiments/026_temporal_mtp.py`](../experiments/026_temporal_mtp.py) is the
main temporal successor:

- context 128;
- hidden size 384, eight layers, six heads;
- predicts offsets 1, 2, 3, 4, 5, 6, 9, 12, 16, and 20;
- a causal temporal decoder conditions later future actions on earlier ones;
- within each frame, later controller groups condition on earlier groups;
- executes a dense four-frame prefix before replanning;
- uses one structured controller codec everywhere.

This architecture directly attacks the coherence problem in 023. A production
026 checkpoint exists and is the reference policy for later IQL work. That does
not by itself prove superhuman strength; it makes 026 the correct modern
baseline to test.

### Experiments 027 through 033

- 027: action-chunk IQL prototype on 026.
- 028: one-hot controller-codec ablation.
- 029: action-conditioned game-state flow expert.
- 030: more paper-faithful double-Q IQL successor.
- 031: frozen-026 Monte Carlo critic probe.
- 032: PPO-ready unified action-token BC decoder on the 026 trunk.
- 033: parallel latent-trajectory policy with best-of-K modes.

These files show the current research direction: coherent action sequences,
better action probability models, critics, and offline RL. Their existence is
not evidence that any one has solved Melee. Each needs recorded closed-loop and
paired results.

## Our Fox/Fox data and latest P0 run

The train filter found:

```text
5,414 Fox/Fox replay rows
10,828 player perspectives
4 windows per replay
21,656 sampled windows per approximate loader pass
about 42.3 batches/pass at batch 512
```

The validation split contains 52 Fox/Fox replay rows. We used 48 fixed
validation samples so the request did not exceed the eligible set.

The run was:

```text
260816-191302_023_mtp_heads_gpt-d256-L8-h4-Lc256-a1024-full-recompute-
o1.5.9.13-linear-chars1v1_ranked-anon-1_p0-fox-ditto
```

Configuration differences from full P0:

- Fox/Fox only;
- 1,024 optimizer steps;
- 32-frame warmup;
- 48 validation samples;
- 32 final Fox-versus-level-9-Fox boots.

At roughly 42.3 batches per corpus pass, 1,024 steps are about 24 passes. The
run therefore did not stop because it saw too little of this dataset.

Results:

```text
final validation action NLL: 1.133
stocks taken/min:            0.832  [0.704, 0.975]
stocks lost/min:             1.632  [1.438, 1.828]
net stocks/min:             -0.800
damage dealt/min:          131.704
damage taken/min:          126.916
net damage/min:              4.788
dead-frame fraction:         0.0235
boots:                      32
matches:                    44
crashed boots:               0
```

Interpretation: the model can enter interactions and accumulate damage, but it
loses stocks much faster than it takes them. Recovery, survival, edge states,
and converting damage into kills remain weak.

Do not compare its stock rates directly with official full-data P0 as an
architecture claim. The opponent schedule, matchup restriction, sample count,
training duration, and validation population differ.

Artifacts:

- W&B: https://wandb.ai/melvinchen/hal/runs/0xsfslbk
- Local checkpoint:
  `runs/260816-191302_023_mtp_heads_gpt-d256-L8-h4-Lc256-a1024-full-recompute-o1.5.9.13-linear-chars1v1_ranked-anon-1_p0-fox-ditto/final.pt`
- R2 prefix:
  `r2:hal/runs/260816-191302_023_mtp_heads_gpt-d256-L8-h4-Lc256-a1024-full-recompute-o1.5.9.13-linear-chars1v1_ranked-anon-1_p0-fox-ditto/`

## Why the policies look bad despite good offline loss

### 1. The objective is imitation, not winning

The loss rewards matching the recorded controller. It does not know that one
choice eventually causes a stock loss unless a value or RL objective supplies
that information.

### 2. Melee actions are multimodal

From one state, good humans may dash back, approach, jump, shield, or wait. A
single categorical model must represent all of those modes. Sampling mixes
styles and intentions from frame to frame.

### 3. P0 has no joint temporal plan

P0 predicts four separate future marginals:

```text
p(a[t+1] | context)
p(a[t+5] | context)
p(a[t+9] | context)
p(a[t+13] | context)
```

It does not model their joint probability. Only `a[t+1]` is executed. The
other heads can improve representation learning, but they do not provide an
executable recovery or combo plan.

### 4. Controller groups are independent in P0

Buttons, main stick, C-stick, and triggers are sampled independently from the
same state. The model can combine pieces that were individually plausible but
not jointly part of one human intent.

### 5. Repeated frames dominate easy metrics

Controller inputs often persist for several frames. Predicting "keep holding
the previous action" gives excellent accuracy on common frames. The model can
therefore lower NLL without improving rare decision boundaries such as:

- the first recovery angle;
- a jump or air-dodge timing;
- ledge options;
- a defensive response after hitlag;
- a kill conversion;
- a change from movement to attack.

P0's official final validation illustrates this. Offset-1 C-stick accuracy was
0.990 because C-stick is usually neutral, not because the model mastered
C-stick decisions.

### 6. Closed-loop errors compound

A human replay never contains the state caused by this model's exact sequence
of earlier mistakes. One bad recovery action creates a new state; the next
prediction is now conditioned on a less familiar trajectory; further errors
accumulate.

The repository's `notebooks/self_nll.py` found a subtle version of this. The
model's own rollouts were lower entropy and easier for it to predict than human
rollouts because ego action-history autocorrelation dominated. This rejects the
simple story that "self-play frames always have higher NLL because they are
OOD." It does not reject compounding control errors. It shows that raw self-NLL
is masked by the shortcut of repeating one's own controls.

### 7. CPU level 9 is not the human training distribution

The opponent has different timing and behavior from ranked humans. The model
can imitate human responses poorly when the opponent generates unfamiliar
sequences. Fox-only training narrows character variation, but it does not make
the CPU behave like a human Fox.

### 8. Data quality is mixed

The ranked corpus includes Platinum, Diamond, and Master labels. Plain P0 gives
their frames equal weight. A larger dataset can still teach a mixture of weak
and strong habits. Rank weighting is an experiment, not an automatic fix,
because it changes the target distribution and can reduce effective data.

### 9. Watching one game is noisy

One rollout is useful for identifying collapse, inactivity, control bugs, or
obvious recovery failures. It is not enough to compare models. Sampling seed,
stage, port, and one early interaction can completely change a game.

## What happened with RL

### PPO

PPO stands for Proximal Policy Optimization. It collects actions from the
current policy, estimates whether each action did better or worse than expected,
and updates action log probabilities while clipping overly large policy
changes.

The value model or critic estimates expected future reward from a state. The
advantage is roughly:

```text
observed return - critic's expected return
```

Positive advantage increases the sampled action's log probability. Negative
advantage decreases it.

Our CPU-9 RL adaptation started from the 009 Fox/Fox behavior-cloned model. The
result collapsed toward repeated up-smash. A rising reward did not mean broad
skill improved; it meant the policy found a behavior that scored well under the
implemented reward and rollout distribution.

Likely contributors, not individually proven causes:

- weak and incoherent warm-start policy;
- reward easier to exploit through damage/attack repetition than through
  long-horizon winning behavior;
- too little opponent and state diversity;
- insufficient anchoring to human behavior;
- action entropy collapsing during optimization;
- CPU-specific exploitation rather than general Melee.

The teacher-anchored RL branch and run watchdogs were attempts to control those
failure modes. They do not turn PPO into a substitute for a strong BC base.

### GRPO and DPO

- GRPO: Group Relative Policy Optimization. It compares several sampled
  outcomes for the same prompt/state and uses their relative rewards. It
  resembles grouped/rejection sampling conceptually, but it still performs a
  policy-gradient update rather than merely retaining winners.
- DPO: Direct Preference Optimization. It learns from preferred versus rejected
  action sequences without an explicit online reward model. It requires a
  meaningful preference dataset and is not a drop-in replacement for game
  interaction.

For Melee, PPO or an offline value method is more natural when exact game
rewards and transitions are available. DPO becomes relevant if we can produce
reliable paired preferences over action chunks or trajectories.

### AWR and IQL

- AWR: Advantage-Weighted Regression. It remains a supervised imitation loss,
  but weights better-than-expected recorded actions more heavily.
- IQL: Implicit Q-Learning. It learns values from offline trajectories and
  improves the policy toward high-value dataset actions without requiring the
  learned policy to generate training rollouts.

These methods can prefer better human actions while staying closer to the data
than online PPO. They still depend on correct reward alignment, critic quality,
and action-sequence modeling.

## Evaluation: what numbers matter

### Offline diagnostics

- action NLL by controller group;
- NLL by prediction horizon;
- hold-frame versus transition-frame NLL;
- action accuracy and calibration;
- ancestor-sampled versus teacher-forced NLL;
- exact-frame action reconstruction;
- action histogram and entropy;
- gradient norms and interaction between primary and auxiliary losses.

An action histogram reports how often the model emits each button combination,
stick cluster, and trigger state. It catches collapse such as neutral-only,
up-smash-heavy, or one-stick-direction policies. It should be compared with a
human validation histogram under the same state distribution.

### Closed-loop diagnostics

- stocks taken and lost per active minute;
- net stocks per minute;
- damage dealt and taken per active minute;
- dead-frame fraction;
- recovery and bottom-blastzone deaths;
- terminal wins/losses;
- crashes and zero-active boots;
- model inference latency.

The primary promotion rule is closed-loop play, not NLL. For a challenger,
mirrored head-to-head against the reference controls for stage and port.

Upstream's fixed evidence rules are:

- 32 deterministic CPU boots for periodic checks;
- 96 boots for a final check;
- 64 mirrored H2H configurations for challenger versus reference;
- bootstrap uncertainty by boot, not by every game produced by one boot;
- keep character schedule, seed, temperature, frame budget, and concurrency
  fixed;
- save match rows, replay files, worker logs, checkpoint hash, and decode
  protocol;
- use at least three paired training seeds for an architecture claim and five
  for a final claim.

### Why bigger reward means "better" only conditionally

A reward is a specification. A larger reward means the policy optimized that
specification more successfully. It means better Melee only if the reward is a
good proxy for winning and cannot be cheaply exploited. Stock lead and winning
are harder to exploit than raw damage, but they are delayed and harder to learn.

## Infrastructure we learned

### Where models and evidence live

Local convention:

```text
runs/<run-name>/latest.pt     resumable checkpoint
runs/<run-name>/final.pt      completed checkpoint
runs/<run-name>/replays/      evaluation replays
```

Cloud convention:

```text
r2:hal/runs/<run-name>/...
```

W&B tracks configuration, metrics, system telemetry, and run identity. A W&B
run page does not guarantee the checkpoint is stored there. HAL's durable
checkpoint store is R2.

Useful commands:

```bash
rclone lsf "r2:hal/runs/<run-name>/" --recursive
rclone copyto "r2:hal/runs/<run-name>/final.pt" "runs/<run-name>/final.pt"
uv run vastai show instances-v1 --raw
uv run vastai logs <instance-id> --tail 200
```

### Vast launcher semantics

`scripts/launch_vast.py` takes the command after `--` and runs that command on
the rented machine. For example:

```bash
uv run scripts/launch_vast.py [launcher options] -- \
  uv run experiments/023_mtp_heads.py [experiment options]
```

`--dry-run` searches and ranks offers without renting one. A normal launch
selects the best ranked qualifying offer and can fail over if an offer cannot
start. Important filters include:

- maximum effective hourly price;
- host RAM;
- GPU VRAM;
- DLPerf;
- CUDA compute capability;
- disk size.

The `ram(GB)` column is host RAM, not GPU memory. A second 5090 offer with twice
the host RAM does not have twice the VRAM or twice the model speed.

`--run-hours` is used for cost estimation and offer ranking. It does not know
the true training duration. The experiment's step limit or an explicit shell
`timeout` determines runtime.

Current launcher protections:

- push and clone the exact source SHA;
- clone from the fork containing that SHA, not an unrelated upstream remote;
- verify CUDA before downloading the large dataset;
- write checkpoints to R2;
- destroy the instance after success;
- destroy it after unattended boot or training failure;
- include a hard command timeout for bounded experiments;
- expose `show instances-v1` to verify that no billing instance remains.

Ctrl-C after an instance is created is not a reliable destroy command. Always
check Vast afterward. Earlier tooling could leave a stopped instance charging
for disk. That, long accidental runtimes, and oversized disks explain prior
cost surprises. The latest Fox P0 run cost about $0.29 and left zero instances.

### The old "reference is not a tree" failure

The launcher tried to clone one remote and check out a SHA that only existed on
another fork. The fix passes `HAL_GIT_REMOTE` with the exact pushed SHA. A public
fork does not require Vast to have a GitHub credential.

### Shell environment detail

`${!k}` is Bash indirect expansion. Zsh interprets `!` differently and produced
`event not found`. In Zsh, use `${(P)k}`, or run the loop under Bash.

### Git topology

The useful setup is:

```text
upstream -> Eric's public repository
origin   -> our fork
```

Start experimental branches from fetched `upstream/main`, then push them to
`origin`. The cloud launcher can clone the public fork and check out the exact
branch SHA. Experiments do not have to be merged into main before launching.

The local `main` once diverged heavily because it represented old fork history.
That did not mean `git pull` could safely guess the desired history. Also,
`git git pull` is simply a typo: the command is `git pull`.

## What is proven, inferred, and unknown

### Proven by current evidence

- Current v7 policy data passed exact full-corpus audits.
- Fox/Fox filtering finds 5,414 train replay rows and 52 validation rows.
- The Fox P0 run completed without crashes and reached NLL 1.133.
- That model loses about 0.80 net stocks per active minute against level-9 Fox.
- Plain P0 is weak despite strong common-frame offline accuracy.
- A larger state-only output MLP did not beat P0 in paired H2H.
- The E2 within-frame factorization run is incomplete.
- The earlier RL policy collapsed to a narrow repeated attack pattern.
- No Vast instance remains active after the latest run.

### Strong interpretations

- Temporal action coherence is a more important next variable than more P0
  epochs.
- Human-action autocorrelation lets one-frame models obtain deceptively good
  offline metrics.
- The weak stock result is consistent with recovery/survival failures rather
  than total inability to interact, because net damage was slightly positive.
- RL should start from a stronger temporal BC policy and retain an explicit
  behavior anchor.

### Still unknown

- Whether 026 trained only on Fox/Fox materially beats 023 under a matched
  protocol.
- Whether the original 009 checkpoint is genuinely stronger or only looked
  better in a few watched games.
- How much performance changes with rank filtering or weighting.
- Which fraction of deaths comes from recovery decisions versus action
  quantization, state features, or opponent distribution.
- Whether 032 or 033 improves closed-loop play over 026.
- Whether an offline critic is accurate enough for IQL/AWR to help.
- Whether better data curation beats a larger model at fixed compute.

## The next experiment that teaches us something

### Question

Does joint temporal action modeling improve Fox/Fox closed-loop play compared
with independent per-frame P0?

### Control

The completed Fox/Fox 023 checkpoint is the control.

### Challenger

Adapt experiment 026 to accept the same ordered `character_pair=(FOX, FOX)`
filter already supported by the shared loader. Keep its native structured
temporal decoder and four-frame execution horizon.

### Hold fixed

- exact train and validation replay eligibility;
- data version and normalization;
- number of approximate corpus passes or total sampled frame positions;
- random seeds;
- CPU level and fixed Fox opponent;
- stage schedule;
- evaluation seed and frame budget;
- sample temperature;
- 32 periodic and 96 final boots;
- replay and metric retention.

### Measurements

- transition-frame NLL, not only total NLL;
- action histograms against human validation;
- recovery/bottom-blastzone death rate;
- stocks taken/lost and net stocks per minute;
- paired H2H 026 versus 023;
- inference speed on local and cloud hardware.

### Decision

Promote temporal modeling only if closed-loop and paired evidence improve. A
lower teacher-forced NLL alone is not enough.

### Do not do yet

- Do not train 023 for another long run.
- Do not launch a broad hyperparameter sweep.
- Do not PPO-fine-tune the weak 023 model.
- Do not change model, data sampler, reward, and evaluation protocol in one run.
- Do not select a checkpoint because one watched game looked good.

## Suggested learning path through the code

### Lesson 1: one replay becomes one model action

Read in this order:

1. [`hal/wire.py`](../hal/wire.py): IDs, buttons, action channels, masks.
2. [`hal/data/schema.py`](../hal/data/schema.py): arrays stored per replay.
3. [`hal/data/extract.py`](../hal/data/extract.py): peppi events to arrays.
4. [`hal/training/dataloader.py`](../hal/training/dataloader.py): replay to
   sampled ego/opp window.
5. [`hal/training/features.py`](../hal/training/features.py): window to tensors.
6. `action_loss` in
   [`experiments/023_mtp_heads.py`](../experiments/023_mtp_heads.py): tensors to
   loss.
7. `make_policy` in the same file: model logits to live actions.
8. [`hal/training/closed_loop.py`](../hal/training/closed_loop.py): rolling live
   context.
9. [`hal/sim/vec.py`](../hal/sim/vec.py): policy actions to Dolphin frames.

Exercise: choose one replay frame and write down the exact value of Fox's
`button_a`, `main_stick_x`, `action`, `position_x`, character ID, and stage ID at
each boundary.

### Lesson 2: understand the baseline mathematically

For P0, trace:

```text
context features
 -> frame token projection
 -> causal transformer
 -> four offset heads
 -> four controller-group logits per offset
 -> cross-entropy against quantized human actions
```

Then trace decode:

```text
last hidden state
 -> offset-1 logits only
 -> categorical sample per controller group
 -> dequantized 14-channel controller
 -> Dolphin
```

The difference between those two traces explains why auxiliary MTP heads are
not themselves a temporal execution policy.

### Lesson 3: understand experiment design

Before every run, write five lines:

```text
Question:
Hypothesis:
Single changed variable:
Primary metric and decision threshold:
What result would falsify the hypothesis:
```

After the run, record:

```text
Exact commit and command:
Checkpoint and hash:
Data and seed:
Offline result:
Closed-loop result:
Decision:
Unexpected observations:
```

That structure prevents "the model looked bad, try something else" from
becoming the research loop.

## Glossary

- Action: one logical controller vector or its quantized representation.
- Action chunk: a sequence of future controller frames.
- Autoregressive: predicts later values conditioned on earlier values.
- BC: behavior cloning, supervised imitation of recorded actions.
- Checkpoint: serialized model, optimizer/config, and training state.
- Closed loop: the model observes consequences of its own earlier actions.
- Context: historical frames visible to the model.
- Critic/value model: estimates expected future return.
- DPO: Direct Preference Optimization.
- Epoch/pass: approximately one traversal of the eligible training corpus.
- Exposure bias: training conditions on true history while deployment receives
  model-generated history.
- GRPO: Group Relative Policy Optimization.
- H2H: head-to-head model comparison.
- IQL: Implicit Q-Learning.
- Log probability: logarithm of the probability assigned to an action.
- MDS: MosaicML Streaming dataset format used for replay samples.
- MTP: multi-token prediction; here, several future action offsets.
- NLL: negative log likelihood; lower means recorded actions were more likely.
- Observation/obs: model-facing recent game state.
- Open loop: executes a planned sequence without replanning from intermediate
  consequences.
- Policy: distribution over actions given observations.
- PPO: Proximal Policy Optimization.
- R2: Cloudflare object storage accessed through an S3-compatible API.
- Receding horizon: predict a chunk, execute part, observe again, and replan.
- Rollout: one policy-driven trajectory in the environment.
- Teacher forcing: train later predictions using recorded earlier actions.
- Temperature: sampling sharpness applied to logits.
- Transition frame: a frame where a controller group changes value.
- W&B: experiment configuration, metrics, and telemetry service.

## Primary source documents in this repository

- [Experiment status](experiments/sequence_status.md)
- [Data pipeline](experiments/data_pipeline.md)
- [Action-chunk roadmap](experiments/action_chunk_roadmap.md)
- [P0/E0 result](experiments/e0_normalized_aux_bc.md)
- [E1 output-head result](experiments/e1_output_head_capacity.md)
- [E2 within-frame factorization](experiments/e2_within_frame_factorization.md)
- [026 closed-loop profile](experiments/026_closed_loop_profile.md)

Use those files for exact historical run IDs, checkpoint hashes, confidence
intervals, and decisions. Use this primer for the overall mental model.
