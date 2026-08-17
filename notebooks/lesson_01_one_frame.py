"""Lesson 1: one replay, one frame, one number.

Answers three questions with real data, no model involved:

  1. What is physically stored for one replay?
  2. What does one frame of "an action" actually look like?
  3. How much does the action change from frame to frame?

Question 3 is the important one. It is the reason a low validation loss has
repeatedly failed to predict closed-loop strength.

Run:
    uv run notebooks/lesson_01_one_frame.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from hal.data.mds import open_shard
from hal.data.mds import read_shard_index
from hal.wire import ACTION_CHANNELS

# The same ranked human corpus the P0 runs train on. Only some shards are local,
# so we read the first one that is actually on disk.
ROOT = Path("data/processed/ranked-anonymized-1/mds")
SPLIT = "train"
# The frame we put under the microscope.
FRAME = 1000


def first_local_shard(shards: list[dict]) -> dict:
    for info in shards:
        raw = ROOT / SPLIT / info["raw_data"]["basename"]
        zipped = info.get("zip_data")
        if raw.is_file() or (zipped and (ROOT / SPLIT / zipped["basename"]).is_file()):
            return info
    raise FileNotFoundError(f"no shard of {ROOT / SPLIT} is present locally")


def action_matrix(sample: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """Stack one player's 14 controller channels into a (T, 14) array."""
    return np.stack([np.asarray(sample[f"{prefix}_{c}"], dtype=np.float32) for c in ACTION_CHANNELS], axis=1)


def show_identity(sample: dict[str, np.ndarray]) -> None:
    frames = len(sample["frame"])
    print(f"frames stored:      {frames}  ({frames / 60:.1f} seconds at 60 fps)")
    print(f"columns stored:     {len(sample)}")
    print(f"schema_version:     {int(np.asarray(sample['schema_version']).reshape(-1)[0])}")
    for name in ("stage", "p1_character", "p2_character"):
        if name in sample:
            value = np.asarray(sample[name]).reshape(-1)[0]
            print(f"{name + ':':<20}{int(value)}")


def show_length_distribution(samples: list[dict[str, np.ndarray]]) -> None:
    """A stored row is one whole game, so lengths vary. Training windows do not."""
    lengths = np.array([len(s["frame"]) for s in samples], dtype=np.int64)
    print("\nHow long is a stored replay row?")
    print(f"    replays:        {len(lengths)}")
    print(f"    distinct lengths: {len(np.unique(lengths))}")
    print(f"    min:            {lengths.min():>7,} frames  ({lengths.min() / 60:>6.1f} s)")
    print(f"    median:         {int(np.median(lengths)):>7,} frames  ({np.median(lengths) / 60:>6.1f} s)")
    print(f"    max:            {lengths.max():>7,} frames  ({lengths.max() / 60:>6.1f} s)")
    print("\n    length histogram (seconds):")
    counts, edges = np.histogram(lengths / 60, bins=10)
    for count, lo, hi in zip(counts, edges[:-1], edges[1:], strict=True):
        print(f"      {lo:>6.0f}-{hi:<6.0f}s {'#' * int(50 * count / counts.max()):<50} {count}")


def show_one_frame(actions: np.ndarray, sample: dict[str, np.ndarray], t: int) -> None:
    print(f"\np1 controller at frame {t} -- this is one 'action', the thing the model predicts:")
    for name, value in zip(ACTION_CHANNELS, actions[t], strict=True):
        marker = "  <- pressed" if value != 0.0 else ""
        print(f"    {name:<16}{value:>8.4f}{marker}")

    print(f"\np1 game state at frame {t} -- part of what the model observes:")
    for suffix in ("action", "position_x", "position_y", "percent", "stock", "airborne", "direction"):
        key = f"p1_{suffix}"
        if key in sample:
            print(f"    {suffix:<16}{float(np.asarray(sample[key])[t]):>8.2f}")


def show_persistence(actions: np.ndarray, t: int, window: int = 8) -> None:
    print(f"\nThe same controller over frames {t}..{t + window - 1}:")
    print(f"    {'frame':<8}{'main_x':>9}{'main_y':>9}{'c_x':>9}{'c_y':>9}{'trig_l':>9}{'A':>4}{'B':>4}{'X':>4}"
          f"{'Y':>4}{'Z':>4}   changed?")
    for i in range(t, t + window):
        a = actions[i]
        changed = "" if i == t else ("CHANGED" if not np.array_equal(a, actions[i - 1]) else "same")
        print(f"    {i:<8}{a[0]:>9.3f}{a[1]:>9.3f}{a[2]:>9.3f}{a[3]:>9.3f}{a[4]:>9.3f}"
              f"{a[6]:>4.0f}{a[7]:>4.0f}{a[8]:>4.0f}{a[9]:>4.0f}{a[10]:>4.0f}   {changed}")


def repeat_baseline(samples: list[dict[str, np.ndarray]]) -> None:
    """The accuracy of the dumbest possible policy: 'output whatever I output last frame'."""
    print("\n" + "=" * 78)
    print("THE NUMBER THAT MATTERS")
    print("=" * 78)

    total = 0
    identical = 0
    per_channel_same = np.zeros(len(ACTION_CHANNELS), dtype=np.int64)
    for sample in samples:
        for prefix in ("p1", "p2"):
            a = action_matrix(sample, prefix)
            if len(a) < 2:
                continue
            same = a[1:] == a[:-1]
            total += len(a) - 1
            identical += int(np.all(same, axis=1).sum())
            per_channel_same += same.sum(axis=0)

    print(f"\nframe transitions examined: {total:,} (across {len(samples)} replays, both players)")
    print(f"\nA policy that only copies its previous frame's controller is exactly")
    print(f"correct on {identical / total:>6.1%} of frames.")
    print("\nPer channel, 'copy the last frame' is correct this often:")
    for name, count in zip(ACTION_CHANNELS, per_channel_same, strict=True):
        frac = count / total
        bar = "#" * int(frac * 50)
        print(f"    {name:<16}{frac:>7.2%}  {bar}")


def main() -> None:
    shards = read_shard_index(ROOT, SPLIT)
    with TemporaryDirectory() as scratch:
        with open_shard(ROOT, SPLIT, first_local_shard(shards), Path(scratch)) as reader:
            samples = [reader[i] for i in range(len(reader))]

    print("=" * 78)
    print("ONE REPLAY")
    print("=" * 78)
    show_identity(samples[0])
    show_length_distribution(samples)

    actions = action_matrix(samples[0], "p1")
    show_one_frame(actions, samples[0], FRAME)
    show_persistence(actions, FRAME)

    repeat_baseline(samples)


if __name__ == "__main__":
    main()
