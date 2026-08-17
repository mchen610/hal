"""Lesson 2: how much of the model's validation NLL is explained by "keep holding it"?

Builds two baselines that contain no neural network and observe no game state.
Both are quantized with the checkpoint's own centers, so they score the exact
quantity experiment 023 logs as ``val/loss``:

    sum over the four action groups of the mean categorical NLL, in bits,
    at offset 1 (the only offset that is actually executed).

  BASELINE A "marginal"  p(a[t+1])            -- ignores everything, even the last frame
  BASELINE B "markov"    p(a[t+1] | a[t])     -- looks only at the previous controller

Baseline B is the honest floor for a Melee policy. Anything a transformer earns
above that floor is what it learned about the *game*.

Run:
    uv run notebooks/lesson_02_dumb_baseline.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from hal.data.mds import open_shard
from hal.data.mds import read_shard_index
from hal.training import scoring
from hal.wire import ACTION_CHANNELS

CKPT = Path(
    "runs/260816-191302_023_mtp_heads_gpt-d256-L8-h4-Lc256-a1024-full-recompute-"
    "o1.5.9.13-linear-chars1v1_ranked-anon-1_p0-fox-ditto/final.pt"
)
ROOT = Path("data/processed/ranked-anonymized-1/mds")
SPLIT = "train"
# The checkpoint trained on Fox/Fox only. Fox is libmelee internal id 1.
CHARACTER_PAIR = (1, 1)
# Offset-1 validation metrics of the checkpoint above, read from its W&B summary
# (melvinchen/hal run 0xsfslbk). Copied in so this script runs without network.
MODEL_OFF1: dict[str, dict[str, float]] = {
    "buttons": {"nll": 0.2559, "hold": 0.0882, "trans": 2.5239, "acc_trans": 0.2503, "pred_change": 0.0289},
    "main_stick": {"nll": 0.7169, "hold": 0.1825, "trans": 3.1356, "acc_trans": 0.3390, "pred_change": 0.1158},
    "c_stick": {"nll": 0.0696, "hold": 0.0184, "trans": 2.9451, "acc_trans": 0.3619, "pred_change": 0.0092},
    "triggers": {"nll": 0.0904, "hold": 0.0210, "trans": 2.6345, "acc_trans": 0.3981, "pred_change": 0.0156},
}
MODEL_VAL_NLL_BITS = sum(v["nll"] for v in MODEL_OFF1.values())
GROUPS = ("buttons", "main_stick", "c_stick", "triggers")
# Laplace smoothing. Without it one unseen transition costs infinite bits.
ALPHA = 0.5
FIT_FRACTION = 0.75
_LN2 = float(np.log(2.0))


def load_centers() -> dict[str, torch.Tensor]:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    return {name: state[name].float() for name in ("main_centers", "c_centers", "trig_centers")}


def group_sizes(centers: dict[str, torch.Tensor]) -> dict[str, int]:
    n_trig = centers["trig_centers"].shape[0]
    return {
        "buttons": scoring.N_BUTTON_COMBOS,
        "main_stick": centers["main_centers"].shape[0],
        "c_stick": centers["c_centers"].shape[0],
        "triggers": n_trig * n_trig,
    }


def quantize(actions: np.ndarray, centers: dict[str, torch.Tensor]) -> np.ndarray:
    """(T, 14) raw controller -> (T, 4) group class indices. Mirrors ``quantize_groups``."""
    a = torch.from_numpy(actions)
    cont, btn = a[:, :6], a[:, 6:]
    n_trig = centers["trig_centers"].shape[0]
    trig = scoring.nearest_center(cont[:, 4:6], centers["trig_centers"])
    idx = torch.stack(
        [
            scoring.buttons_to_combo(btn),
            scoring.nearest_cluster(cont[:, 0:2], centers["main_centers"]),
            scoring.nearest_cluster(cont[:, 2:4], centers["c_centers"]),
            trig[:, 0] * n_trig + trig[:, 1],
        ],
        dim=-1,
    )
    return idx.numpy().astype(np.int64)


def replay_actions(sample: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return np.stack([np.asarray(sample[f"{prefix}_{c}"], dtype=np.float32) for c in ACTION_CHANNELS], axis=1)


def load_quantized(pair: tuple[int, int] | None, centers: dict[str, torch.Tensor]) -> list[list[np.ndarray]]:
    """Per matching replay, the (T, 4) index array of each player perspective.

    Grouped by replay, not flattened: p1 and p2 of one game watch the same states,
    so they must land on the same side of the fit/held-out split.
    """
    shards = read_shard_index(ROOT, SPLIT)
    info = next(
        s
        for s in shards
        if (ROOT / SPLIT / s["raw_data"]["basename"]).is_file()
        or (s.get("zip_data") and (ROOT / SPLIT / s["zip_data"]["basename"]).is_file())
    )
    out: list[list[np.ndarray]] = []
    with TemporaryDirectory() as scratch:
        with open_shard(ROOT, SPLIT, info, Path(scratch)) as reader:
            for i in range(len(reader)):
                sample = reader[i]
                chars = tuple(int(np.asarray(sample[f"p{p}_character"]).reshape(-1)[0]) for p in (1, 2))
                if pair is not None and chars != pair:
                    continue
                out.append([quantize(replay_actions(sample, prefix), centers) for prefix in ("p1", "p2")])
    return out


def fit_counts(episodes: list[np.ndarray], sizes: dict[str, int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per group: (transition counts [V, V], marginal counts [V])."""
    counts = {g: (np.zeros((sizes[g], sizes[g]), np.float64), np.zeros(sizes[g], np.float64)) for g in GROUPS}
    for episode in episodes:
        for g_i, g in enumerate(GROUPS):
            q = episode[:, g_i]
            trans, marg = counts[g]
            np.add.at(trans, (q[:-1], q[1:]), 1.0)
            np.add.at(marg, q[1:], 1.0)
    return counts


def score(
    episodes: list[np.ndarray],
    counts: dict[str, tuple[np.ndarray, np.ndarray]],
    sizes: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Mean NLL in bits per group, for both baselines, on held-out episodes."""
    out: dict[str, dict[str, float]] = {}
    for g_i, g in enumerate(GROUPS):
        trans, marg = counts[g]
        v = sizes[g]
        log_p_markov = np.log((trans + ALPHA) / (trans.sum(axis=1, keepdims=True) + ALPHA * v))
        log_p_marginal = np.log((marg + ALPHA) / (marg.sum() + ALPHA * v))

        totals = {"markov": 0.0, "marginal": 0.0, "markov_trans": 0.0, "markov_hold": 0.0}
        n = n_trans = 0
        for episode in episodes:
            q = episode[:, g_i]
            bits = -log_p_markov[q[:-1], q[1:]]
            # 023's definition: frame t is a transition iff the target q[t+1] differs from q[t].
            is_trans = q[1:] != q[:-1]
            totals["markov"] += float(bits.sum())
            totals["marginal"] += float(-log_p_marginal[q[1:]].sum())
            totals["markov_trans"] += float(bits[is_trans].sum())
            totals["markov_hold"] += float(bits[~is_trans].sum())
            n += len(q) - 1
            n_trans += int(is_trans.sum())
        out[g] = {
            "markov": totals["markov"] / n / _LN2,
            "marginal": totals["marginal"] / n / _LN2,
            "markov_trans": totals["markov_trans"] / max(n_trans, 1) / _LN2,
            "markov_hold": totals["markov_hold"] / max(n - n_trans, 1) / _LN2,
            "trans_rate": n_trans / n,
            "frames": float(n),
        }
    return out


def report(title: str, results: dict[str, dict[str, float]], n_fit: int, n_eval: int) -> float:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"fit on {n_fit} perspectives, scored on {n_eval} held-out perspectives "
          f"({int(results['buttons']['frames']):,} frames)")
    print(f"\n{'group':<14}{'marginal':>10}{'markov':>10}   |{'% frames':>10}{'markov on':>11}{'markov on':>11}")
    print(f"{'':<14}{'':>10}{'':>10}   |{'changing':>10}{'CHANGE':>11}{'HOLD':>11}")
    for g in GROUPS:
        r = results[g]
        print(f"{g:<14}{r['marginal']:>10.4f}{r['markov']:>10.4f}   |{r['trans_rate']:>10.2%}"
              f"{r['markov_trans']:>11.3f}{r['markov_hold']:>11.3f}")
    markov = sum(results[g]["markov"] for g in GROUPS)
    marginal = sum(results[g]["marginal"] for g in GROUPS)
    print(f"{'-' * 68}")
    print(f"{'TOTAL (bits)':<14}{marginal:>10.4f}{markov:>10.4f}")
    return markov


def main() -> None:
    centers = load_centers()
    sizes = group_sizes(centers)
    print(f"quantization from the checkpoint: {sizes}")

    replays = load_quantized(CHARACTER_PAIR, centers)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(replays))
    cut = int(len(order) * FIT_FRACTION)
    # Split by replay so no game contributes one perspective to fit and the other to held-out.
    fit = [e for i in order[:cut] for e in replays[i]]
    held = [e for i in order[cut:] for e in replays[i]]
    print(f"{len(replays)} replays -> {len(order[:cut])} fit / {len(order[cut:])} held-out (split by game)")

    counts = fit_counts(fit, sizes)
    results = score(held, counts, sizes)
    markov = report("FOX/FOX -- same matchup the checkpoint trained on", results, len(fit), len(held))

    print("\n" + "=" * 78)
    print("MODEL vs BASELINE, split by frame type")
    print("=" * 78)
    print(f"\n{'':<14}{'HOLD frames (bits)':>26}{'CHANGE frames (bits)':>28}")
    print(f"{'group':<14}{'markov':>12}{'model':>7}{'gain':>7}{'markov':>12}{'model':>8}{'gain':>8}")
    for g in GROUPS:
        b, m = results[g], MODEL_OFF1[g]
        print(f"{g:<14}{b['markov_hold']:>12.3f}{m['hold']:>7.3f}{b['markov_hold'] - m['hold']:>7.3f}"
              f"{b['markov_trans']:>12.3f}{m['trans']:>8.3f}{b['markov_trans'] - m['trans']:>8.3f}")

    print("\nOn the frames that decide games, how often is the model exactly right?")
    for g in GROUPS:
        print(f"    {g:<14}{MODEL_OFF1[g]['acc_trans']:>7.1%} on CHANGE frames "
              f"(true change rate here: {results[g]['trans_rate']:.1%})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  trained transformer (023 P0, Fox/Fox):        {MODEL_VAL_NLL_BITS:.3f} bits/frame")
    print(f"  markov baseline, previous controller only:    {markov:.3f} bits/frame")
    print(f"  marginal baseline, no context at all:         {sum(results[g]['marginal'] for g in GROUPS):.3f}"
          " bits/frame")
    delta = MODEL_VAL_NLL_BITS - markov
    verdict = "the transformer is WORSE than copy-with-statistics by" if delta > 0 else "the transformer beats it by"
    print(f"\n  {verdict} {abs(delta):.3f} bits/frame")


if __name__ == "__main__":
    main()
