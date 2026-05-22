"""Eval the trained model across stages in two modes:

  * "vs-cpu"    — model on port 1 vs in-game lvl-9 CPU Fox on port 2.
  * "self-play" — model on both ports, ONE batched forward (batch_dim=2)
                  per chunk boundary inside SelfPlayController.

Both modes run on the exi-ai Ishiiruka build with EXI inputs + FFW + uncapped
emulation (the WX-GUI slippi build does not render in this headless dev env).

Run:
    DISPLAY=:0 .venv/bin/python notebooks/eval_vs_cpu.py
"""

import os
import time
import urllib.parse
from pathlib import Path

import numpy as np
import torch
from toy_train import FlowMatchingPolicy
from toy_train import ModelControllerSource
from toy_train import SelfPlayController
from toy_train import stats

CKPT_DEFAULT = "runs/260521-111610_fm-d256-L6-H8-Lc256-Lk16-fs8_ranked-anon-1_hist-dropout-2k/final.pt"
# Long enough that a lvl-9 CPU finishes our 4 stocks (or vice versa in self-play)
# so Dolphin writes a real GameEnd footer — peppi-py refuses to parse otherwise.
# Safety cap — Instant Match is OFF, so drive() returns the moment a match
# ends naturally (GameEnd → menu → in_game=False). The cap only matters if
# the model + CPU stalemate run the in-game timer to ~28000 frames.
MAX_FRAMES = 15_000

EXIAI_DOLPHIN = os.environ.get("HAL_EMULATOR_PATH", "/home/ericgu/src/hal/data/emulator/exiai/squashfs-root/AppRun")

SLIPPILAB_URL = "http://localhost:5173"
SLIPPILAB_PUBLIC = Path("~/src/slippilab/public").expanduser()
SERVE_MOUNT = "hal-replays"
SERVE_DIR = SLIPPILAB_PUBLIC / SERVE_MOUNT

REPLAY_DIR = (Path(__file__).resolve().parent / ".filter_explore" / "extracted").resolve()
REPLAY_DIR.mkdir(parents=True, exist_ok=True)
if not SERVE_DIR.exists() and SLIPPILAB_PUBLIC.exists():
    SERVE_DIR.symlink_to(REPLAY_DIR)


MODES: tuple[str, ...] = ("vs-cpu", "self-play")


def _last_finite(arr: np.ndarray) -> float:
    """Last entry of `arr` that isn't NaN. Trailing frames captured during the
    IN_GAME → menu transition have NaN per-port fields; we want the last
    in-game value."""
    finite = arr[np.isfinite(arr)]
    return float(finite[-1]) if len(finite) > 0 else 0.0


def _take_first_match_slp(replay_dir: Path, before: set[str], match_start: float) -> Path | None:
    """The first .slp Dolphin wrote during this Session — the only one with
    a guaranteed GameEnd footer (subsequent matches auto-restart via the
    Instant Match gecko code; the last in-progress one gets truncated when
    Session shuts Dolphin down)."""
    new = [p for p in replay_dir.glob("*.slp") if p.name not in before and p.stat().st_mtime >= match_start]
    if not new:
        return None
    new.sort(key=lambda p: p.stat().st_mtime)
    return new[0]


def _build_matchup_and_sources(mode: str, stage, model, device: str):
    import melee

    from hal.sim.session import Matchup
    from hal.sim.session import PlayerSetup
    from hal.sim.sources import InternalControllerSource

    if mode == "vs-cpu":
        matchup = Matchup(
            stage=stage,
            players=(
                PlayerSetup(port=1, character=melee.Character.FOX, cpu_level=0),
                PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=9),
            ),
        )
        sources = {
            1: ModelControllerSource(model=model, stats=stats, ego_prefix="p1", device=device),
            2: InternalControllerSource(),
        }
        return matchup, sources
    coord = SelfPlayController(model=model, stats=stats, device=device)
    matchup = Matchup(
        stage=stage,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX, cpu_level=0),
            PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=0),
        ),
    )
    sources = {1: coord.view("p1"), 2: coord.view("p2")}
    return matchup, sources


def main() -> None:
    import melee

    from hal.fixtures import DOLPHIN_EXIAI
    from hal.fixtures import ISO
    from hal.fixtures import ensure
    from hal.policy import INCLUDED_STAGES
    from hal.sim.loop import drive
    from hal.sim.session import Session

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.environ.get("CKPT", CKPT_DEFAULT)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    model = FlowMatchingPolicy(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[eval] loaded {ckpt_path}  step={state['step']}  device={device}", flush=True)

    ensure(DOLPHIN_EXIAI)
    iso = ensure(ISO)

    stages = [s for s in INCLUDED_STAGES if s is not melee.Stage.FOUNTAIN_OF_DREAMS]

    results: list[tuple[str, str, Path, dict]] = []
    for mode in MODES:
        print(f"\n[eval] ============== mode={mode} ==============", flush=True)
        for stage in stages:
            before = {p.name for p in REPLAY_DIR.glob("*.slp")}
            matchup, sources = _build_matchup_and_sources(mode, stage, model, device)
            print(f"[eval] === mode={mode}  stage={stage.name} ===", flush=True)
            t0 = time.monotonic()
            try:
                with Session(
                    iso_path=iso,
                    dolphin_path=EXIAI_DOLPHIN,
                    blocking_input=True,
                    tmp_home_directory=True,
                    replay_dir=str(REPLAY_DIR),
                    step_timeout_seconds=30.0,
                    use_exi_inputs=True,
                    enable_ffw=True,
                    emulation_speed=0.0,
                ) as s:
                    traj = drive(s, matchup, sources, max_frames=MAX_FRAMES)
            except Exception as e:
                print(f"[eval] FAILED mode={mode} stage={stage.name}: {e!r}", flush=True)
                continue
            wall = time.monotonic() - t0
            slp = _take_first_match_slp(REPLAY_DIR, before, t0)
            # Last frame is often a menu transition with NaN per-port fields.
            # Walk backward to the last in-game frame.
            p1_stock = int(_last_finite(traj.post[1]["stock"]))
            p2_stock = int(_last_finite(traj.post[2]["stock"]))
            p1_max_pct = float(np.nanmax(traj.post[1]["percent"]))
            p2_max_pct = float(np.nanmax(traj.post[2]["percent"]))
            summary = dict(
                frames=len(traj),
                p1_stocks_left=p1_stock,
                p2_stocks_left=p2_stock,
                p1_max_pct=p1_max_pct,
                p2_max_pct=p2_max_pct,
                wall_s=wall,
            )
            print(f"[eval] {mode}/{stage.name}: {summary}  slp={slp.name if slp else 'NONE'}", flush=True)
            if slp is None:
                continue
            renamed = slp.with_name(f"hal-{mode}-{stage.name.lower()}-{slp.stem}.slp")
            slp.rename(renamed)
            results.append((mode, stage.name, renamed, summary))

    print("\n\n=== Slippilab URLs ===\n")
    for mode, stage_name, slp_path, summary in results:
        replay_url = f"{SLIPPILAB_URL}/{SERVE_MOUNT}/{slp_path.name}"
        link = f"{SLIPPILAB_URL}/?replayUrl={urllib.parse.quote(replay_url, safe=':/')}"
        print(
            f"  [{mode:>10s}] {stage_name:18s}  "
            f"stocks_left p1={summary['p1_stocks_left']} p2={summary['p2_stocks_left']}  "
            f"max_pct p1={summary['p1_max_pct']:5.1f} p2={summary['p2_max_pct']:5.1f}  "
            f"frames={summary['frames']}  wall={summary['wall_s']:.1f}s"
        )
        print(f"    {link}")


if __name__ == "__main__":
    main()
