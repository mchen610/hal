# Replay reproduction — handoff

Forward-looking summary. Pair with the README (current state) and
`replay_reproduction_sanity.repro_log.md` (process / dead ends).

## Where we are

`notebooks/replay_reproduction_sanity.py` takes a `.slp`, replays its
controller inputs through libmelee → Dolphin, and asserts the live
game state matches the recorded state frame-by-frame. Two run modes:

- `normal` — windowed Dolphin, no EXI override.
- `ffw` — headless Null gfx + `Allow Bot Input Overrides` EXI gecko.

After the recent fixes the harness is **bit-exact when the source slp's
recording build matches the live emulator's build**. Verified by
self-feed: re-fed a slp the live emulator just wrote (slp 3.19.0,
exi-ai-rebase) back as source — 170 frames, every comparison field,
zero mismatches.

When source is the dev replay (slp 3.7.0, recorded ~2020-12 with
slippi-Ishiiruka v2.2.3), 21 of 300 frames mismatch — *all
`hitlag_left`*, on hit moments, decaying to 0 within ~7 frames. Caused
by build-version drift (UCF 0.74 in v2.2.3 vs UCF 0.84's new
`Pad Buffer + 1.0 Cardinals` patch in current). Not closable on the
harness side; needs a v2.2.3-era nogui binary.

## What got us here

Two non-obvious fixes worth knowing:

1. **Mode-aware trigger inverse.** Normal and FFW go through different
   Dolphin paths, so the same pipe value can't satisfy both:
   - Normal reads triggers via `Triggers/L-Analog = Axis L +`. The
     pipe `value` must satisfy `max(0, value - 0.5) * 2 ≈ amount`,
     which means `pipe ≈ amount/2 + 0.5` (always ≥ 0.5).
   - FFW reads `padBuf[6]` directly via `prepareOverwriteInputs`, so
     the pipe must satisfy `u8(value * 255) = round(amount * 0x8C)`,
     which means `pipe ≈ raw/255` (often < 0.5).
   - The two are incompatible. Sender picks based on `use_exi_inputs`.
   - The previous code used the FFW formula in both modes, so normal
     mode triggered `Axis L + = 0` and live read trigger = 0. That
     showed up as the `controller.l_shoulder` mismatch from frame -4.

2. **`blocking_input=True`.** Pulled from libmelee's own `test_live.py`
   canonical config. Sets Dolphin's `m_blockingPipes`, which makes the
   input pipe `select()` until the bot writes a complete `FLUSH`
   frame. Without it, FFW's EXI overwrite path could read stale
   `padBuf` between our writes — that was the FFW main-stick
   staleness symptom from frame ~64.

## Followup tests / debugging

In rough priority order:

1. **Self-feed bit-exact regression test.** Run the harness with
   `save_replays=True`, finalize the slp (the live emulator stops
   mid-game so the file is unfinalized), re-feed as source, assert 0
   mismatches. The trailer patch:
   ```python
   raw_length = bytes_written - raw_data_offset
   data[length_offset:length_offset+4] = struct.pack('>I', raw_length)
   data.extend(b'U\x08metadata{}}')  # close raw, append empty metadata, close outer
   ```
   This catches harness regressions without depending on external
   replays or matching slip versions.

2. **Mode-equivalence test.** On a self-fed slp, `normal` and `ffw`
   should both be zero-mismatch. Currently both have the same residual
   on the dev replay (so that's the right sanity), but the stronger
   property is "modes agree on a same-build slp."

3. **Stick byte calibration sweep.** Pre-game, sweep main X across
   `[-1, 1]` in fine steps, record live's slp 0x19 (processed) and
   0x3B (raw byte) per frame. Saves a CSV that documents the live
   build's exact byte→processed mapping. Useful for future
   investigations and for cross-build A/B if someone builds an older
   nogui binary.

4. **Build-version compatibility check at harness startup.** Compare
   `source_slp_version_tuple` to `console.dolphin_version` and warn
   on mismatch (see `Version compatibility` below).

5. **Trigger inverse integration verification.** For amounts in
   `{0.0, 0.05, 0.1, …, 1.0}` in both modes, run a frame, verify
   live's slp 0x29 records the input amount. Currently we have a unit
   test that exercises the math through Dolphin's quantization in
   pure Python; an integration variant catches build behavior changes.

6. **Build a v2.2.3 nogui binary.** `~/src/slippi-Ishiiruka` is checked
   out; `git checkout v2.2.3 && ./build-linux.sh` plus dolphin's full
   build deps. Run the harness against it on the dev replay; expect 0
   hitlag mismatches. Definitively confirms the build-drift
   hypothesis. Not blocking, but closes the loop.

## Developer-facing API plan

Today's harness mixes I/O (CLI / JSONL), Dolphin orchestration, and
frame comparison. Three layers, separate concerns:

### Layer 1: emulator session

Owns a live Dolphin process; sends controller inputs frame-by-frame;
yields game states.

```python
class EmulatorSession:
    def __init__(self, mode: Literal["normal", "ffw"], iso_path, source_metadata): ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *_): ...
    def send_inputs(self, by_port: dict[int, melee.ControllerState]) -> None: ...
    def step(self) -> melee.GameState | None: ...
    def slp_path(self) -> Path | None: ...   # if save_replays
```

Encapsulates the per-mode kwargs, menu navigation, and the
`blocking_input=True` / `polling_mode=True` config.

### Layer 2: replay driver

Replays a source's frames through a session; supports an optional
controller callback so a model can override inputs.

```python
def drive_session(
    session: EmulatorSession,
    source_frames: dict[int, FrameRecord],
    *,
    controller_callback: Callable[[melee.GameState, int], dict[int, melee.ControllerState]] | None = None,
    start_frame: int = 0,
    n_frames: int = 300,
) -> Iterator[tuple[FrameRecord, melee.GameState]]:
    """Yields (source_record, live_state) per frame. If callback is None, sends
    source's own inputs (bit-exact reproduction). Otherwise, sends callback's
    outputs (model evaluation)."""
```

### Layer 3: developer entrypoints

Two top-level helpers cover the two real workflows.

```python
def verify_replay_reproduction(
    replay_path: Path,
    *,
    mode: Literal["normal", "ffw"] = "normal",
    prefix_frames: int = 300,
    start_frame: int = 0,
    fields: tuple[str, ...] = ALL_FIELDS,
    fail_on_mismatch: bool = True,
) -> ReproductionResult:
    """Replay source's inputs back through live; compare frame-by-frame."""
```

```python
def evaluate_model_against_replay(
    model: Callable[[melee.GameState, int], dict[int, melee.ControllerState]],
    replay_path: Path,
    *,
    mode: Literal["normal", "ffw"] = "ffw",
    n_frames: int = 300,
    record_live_slp: bool = True,
) -> EvaluationResult:
    """Run model against a live emulator booted from the replay's start. Returns
    per-frame states + path to the live slp the emulator wrote (if enabled)."""
```

For HAL specifically: `evaluate_model_against_replay(model, ...)` is
the natural plug-point. The model gets `(GameState, frame_index)`,
returns the same `ControllerState` dict shape that source replays
produce. No HAL-specific preprocessing or postprocessing in the
reproduction layer — the model handles its own discretization /
decoding.

### File-system layout

Move production code out of `notebooks/`:

```
hal/replay_reproduction/
  __init__.py            -- re-exports the entrypoints
  sender.py              -- pipe quantization helpers + ReplayControllerSender
  session.py             -- EmulatorSession
  comparator.py          -- compare_states, mismatch dataclasses
  driver.py              -- drive_session, verify_replay_reproduction
  evaluation.py          -- evaluate_model_against_replay
  compat.py              -- version compatibility checks (see below)
  cli.py                 -- python -m hal.replay_reproduction <replay>

hal/test_replay_reproduction.py   -- unit + gated integration tests
notebooks/replay_reproduction_sanity.py    -- thin demo importing from hal.replay_reproduction
```

The README, repro_log, and HANDOFF docs can stay under `notebooks/` as
companion docs to the demo, or move to `hal/replay_reproduction/docs/`.

## Version compatibility

Three independent versions need to align for bit-exact reproduction:

- **slp version** — written by the recording build (e.g. `3.7.0`).
- **slippi-Ishiiruka build** — the live emulator's release / commit.
- **libmelee version** — the Python package's parser.

Within libmelee's supported range, slp version 1 ↔ build version is
the load-bearing axis: the build's gecko codeset (`Sys/GameSettings/
GALE01r2.ini`) controls UCF, hitlag patches, byte-level stick
processing, etc. Different builds → different game state for the same
input bytes.

### Programmatic compatibility check

Add `hal/replay_reproduction/compat.py`:

```python
@dataclass
class CompatResult:
    status: Literal["ok", "warn", "incompatible"]
    diagnostic: str

def check_replay_emulator_compat(
    source_slp_version: tuple[int, int, int],
    live_dolphin_version: melee.console.DolphinVersion,
) -> CompatResult: ...
```

Wire it into `verify_replay_reproduction` and
`evaluate_model_against_replay`. Default behavior:

- `ok`: same compat class → silent pass.
- `warn`: same UCF major version, different minor build → log
  `expect minor hitlag drift on hit moments; bit-exact for action /
  position / processed controller / shield / hitstun.`
- `incompatible`: different UCF major (0.74 vs 0.84) or older slp →
  raise unless `allow_version_drift=True`.

### Compat matrix (initial; needs verification)

Filled in from what we observed; the slippi-Ishiiruka tag → ini
mapping needs cross-checking against release tags' `GALE01r2.ini`.

| slp version range | Recorded under (estimate) | UCF | Notable patches | Notes |
|---|---|---|---|---|
| < 1.2.0 | < v1.4.0 | none | — | No raw stick byte. libmelee `allow_old_version=True` required. |
| 1.2.0 – 2.x | v1.4.x – v1.x | 0.74 | Raw main X (0x3B). | |
| 3.0.0 – 3.6.0 | v2.0 – v2.1 | 0.74 | Frame Start (0x3A). | |
| 3.7.0 – 3.14.x | v2.2.x – v2.4.x | 0.74 → 0.84 transition | Pad Buffer + 1.0 Cardinals lands somewhere here. | **Dev replay's range.** Hitlag drift vs UCF-0.84 emulators. |
| 3.15.0 – 3.16.x | v2.5 – v3.0 | 0.84 | Raw main Y (0x40). | |
| 3.17.0 – 3.18.x | v3.0 – v3.x | 0.84 | C-stick raw (0x41/0x42). | |
| 3.19.0+ | exi-ai-rebase / v3.x | 0.84 + Pad Buffer | What current emulator writes. | Bit-exact reproducible by current emulator. |

### Recommendations

For zero version drift in HAL:

1. **For new datasets:** record under the current emulator. Even a
   bot-vs-bot or human-vs-bot mini-corpus avoids drift entirely.
2. **For existing 3.7-era datasets (the dev / top-player slps):**
   accept the documented hitlag-only residual. It does not affect
   any feature your current input/target configs actually use
   (processed sticks + analog shoulders). Train as is.
3. **If a future dataset / config introduces byte-level features
   (e.g. raw stick byte input):** version drift starts to matter.
   Either re-record or build a matching nogui slippi-Ishiiruka.

## Cleanup checklist

- [ ] Carve `notebooks/replay_reproduction_sanity.py` into
  `hal/replay_reproduction/` modules (Layer 1/2/3 above).
- [ ] Add `compat.check_replay_emulator_compat` + wire warnings into
  `verify_replay_reproduction` and `evaluate_model_against_replay`.
- [ ] Add the self-feed regression test (followup #1).
- [ ] Add `evaluate_model_against_replay` as the primary HAL entrypoint
  for closed-loop testing of a trained model.
- [ ] Replace `notebooks/replay_reproduction_sanity.py` with a thin
  CLI shim importing from `hal.replay_reproduction`.
- [ ] Document the compat matrix in HAL's main README under a
  "Closed-loop evaluation" section, alongside the developer workflow.
- [ ] Optional: assert live's saved slp byte-matches source's slp on
  matching-build runs (stricter than per-field comparison).
