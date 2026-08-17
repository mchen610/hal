"""Lesson 3: what "the four groups are independent" costs, measured on the real checkpoint.

P0 predicts four separate distributions from one hidden state --- buttons, main
stick, C-stick, triggers --- and samples each one on its own. The joint
probability it assigns to a controller is therefore the product of four marginals:

    p(action) = p(buttons) * p(main) * p(c) * p(triggers)

A human controller is not built that way. "Shine" is B *with* the stick neutral;
"up-smash" is A *with* the stick up. This script measures how often independent
sampling assembles a controller no human ever produced.

Run:
    uv run notebooks/lesson_03_independence.py
"""

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from hal.data.mds import open_shard
from hal.data.mds import read_shard_index
from hal.training.dataloader import collate_train_batch
from hal.training.dataloader import relabel_ego
from hal.wire import ACTION_CHANNELS

CKPT = (
    "runs/260816-191302_023_mtp_heads_gpt-d256-L8-h4-Lc256-a1024-full-recompute-"
    "o1.5.9.13-linear-chars1v1_ranked-anon-1_p0-fox-ditto/final.pt"
)
ROOT = Path("data/processed/ranked-anonymized-1/mds")
SPLIT = "train"
CHARACTER_PAIR = (1, 1)
N_WINDOWS = 48
SEED = 0
BUTTON_NAMES = tuple(c.removeprefix("button_") for c in ACTION_CHANNELS[6:])


def load_experiment():
    """Import experiments/023_mtp_heads.py by path (its filename starts with a digit)."""
    spec = importlib.util.spec_from_file_location("exp023", "experiments/023_mtp_heads.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp023"] = module
    spec.loader.exec_module(module)
    return module


def load_replays(exp) -> list[dict]:
    shards = read_shard_index(ROOT, SPLIT)
    info = next(
        s
        for s in shards
        if (ROOT / SPLIT / s["raw_data"]["basename"]).is_file()
        or (s.get("zip_data") and (ROOT / SPLIT / s["zip_data"]["basename"]).is_file())
    )
    keep: list[dict] = []
    with TemporaryDirectory() as scratch:
        with open_shard(ROOT, SPLIT, info, Path(scratch)) as reader:
            for i in range(len(reader)):
                sample = reader[i]
                chars = tuple(int(np.asarray(sample[f"p{p}_character"]).reshape(-1)[0]) for p in (1, 2))
                if chars == CHARACTER_PAIR:
                    keep.append({k: np.asarray(v) for k, v in sample.items()})
    return keep


def make_batch(exp, replays: list[dict], cfg, stats, rng: np.random.Generator):
    """Build one TrainBatch of real windows, exactly as the trainer's collate does."""
    seq = cfg.L_ctx + max(cfg.head_offsets)
    windows = []
    for _ in range(N_WINDOWS):
        sample = replays[rng.integers(len(replays))]
        frames = len(sample["frame"])
        start = int(rng.integers(0, frames - seq))
        ego = "p1" if rng.integers(2) == 0 else "p2"
        window = relabel_ego({k: v[start : start + seq] for k, v in sample.items() if v.ndim >= 1}, ego)
        window = {k: v for k, v in window.items() if len(v) == seq}
        window["ctx_pad"] = np.int64(0)
        windows.append(window)
    return collate_train_batch(windows, stats=stats, L_ctx=cfg.L_ctx, projection=exp._INPUT_PROJECTION)


def describe_buttons(combo: int) -> str:
    bits = [BUTTON_NAMES[i] for i in range(8) if combo >> i & 1]
    return "+".join(bits) if bits else "(none)"


def show_one_frame(exp, model, batch, centers) -> None:
    """Print the four marginals the model produces at a single real frame."""
    with torch.no_grad():
        h = model(batch.context.features, batch.context.ctx_pad)
        logits = model.group_logits(h[:, -1:], head_index=0)
    print("\n" + "=" * 78)
    print("THE FOUR MARGINALS AT ONE REAL FRAME")
    print("=" * 78)
    for name in exp._GROUP_NAMES:
        p = torch.softmax(logits[name][0, 0].float(), dim=-1)
        top = torch.topk(p, 4)
        entropy = float(-(p * p.clamp_min(1e-12).log2()).sum())
        print(f"\n  {name}  (entropy {entropy:.2f} bits)")
        for prob, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True):
            label = describe_buttons(idx) if name == "buttons" else _label(name, idx, centers)
            print(f"      {prob:>6.1%}  {label}")


def _label(name: str, idx: int, centers: dict[str, torch.Tensor]) -> str:
    if name == "main_stick":
        x, y = centers["main_centers"][idx].tolist()
        return f"stick ({x:+.2f}, {y:+.2f})"
    if name == "c_stick":
        x, y = centers["c_centers"][idx].tolist()
        return f"c-stick ({x:+.2f}, {y:+.2f})"
    n = centers["trig_centers"].shape[0]
    return f"triggers (L={centers['trig_centers'][idx // n]:.2f}, R={centers['trig_centers'][idx % n]:.2f})"


def joint_support(exp, replays: list[dict], model) -> set[tuple[int, int, int, int]]:
    """Every joint (buttons, main, c, triggers) combination humans actually produced."""
    seen: set[tuple[int, int, int, int]] = set()
    for sample in replays:
        for prefix in ("p1", "p2"):
            actions = np.stack(
                [np.asarray(sample[f"{prefix}_{c}"], dtype=np.float32) for c in ACTION_CHANNELS], axis=1
            )
            idx = exp._quantize(model, torch.from_numpy(actions))
            seen.update(map(tuple, idx.tolist()))
    return seen


def main() -> None:
    exp = load_experiment()
    model, cfg, stats, _ = exp._load_ckpt(CKPT)
    model.eval()
    # The run trained on a GPU with FlexAttention required. This lesson runs on CPU,
    # where the trunk falls back to SDPA. Same maths, different kernel.
    model.trunk.require_flex = False
    centers = {n: getattr(model, n).float() for n in ("main_centers", "c_centers", "trig_centers")}
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    replays = load_replays(exp)
    print(f"{len(replays)} Fox/Fox replays; device={exp.DEVICE}; L_ctx={cfg.L_ctx}")

    batch = make_batch(exp, replays, cfg, stats, rng).to(exp.DEVICE)
    show_one_frame(exp, model, batch, centers)

    # Every joint controller humans produced in this corpus.
    support = joint_support(exp, replays, model)
    print(f"\nhuman Fox/Fox joint controllers ever observed: {len(support):,} "
          f"of {256 * 65 * 9 * 25:,} representable")

    # Sample one action per frame from the model's independent product, at every
    # context position of every window, and ask whether a human ever produced it.
    with torch.no_grad():
        h = model(batch.context.features, batch.context.ctx_pad)
        logits = model.group_logits(h, head_index=0)
        probs = {g: torch.softmax(logits[g].float(), dim=-1) for g in exp._GROUP_NAMES}
        sampled = torch.stack(
            [torch.multinomial(probs[g].reshape(-1, probs[g].shape[-1]), 1).squeeze(-1) for g in exp._GROUP_NAMES],
            dim=-1,
        )
    B, L = h.shape[0], h.shape[1]
    # Position t predicts t+1. Line all three up on positions 0 .. L-2.
    truth = exp._quantize(model, exp.stack_actions(batch.context.features)).cpu().numpy()  # [B, L, 4]
    held = truth[:, :-1].reshape(-1, 4)  # what the player is doing at t
    human_next = truth[:, 1:].reshape(-1, 4)  # what the human did at t+1
    model_next = sampled.reshape(B, L, 4)[:, :-1].reshape(-1, 4).cpu().numpy()  # model's sample for t+1

    sampled_idx = model_next
    model_unseen = sum(1 for row in sampled_idx if tuple(row) not in support)
    human_unseen = sum(1 for row in human_next if tuple(row) not in support)
    n = len(sampled_idx)

    print("\n" + "=" * 78)
    print("HOW OFTEN DOES INDEPENDENT SAMPLING INVENT A CONTROLLER NO HUMAN MADE?")
    print("=" * 78)
    print(f"\n  frames sampled:                                    {n:,}")
    print(f"  model samples never seen in human Fox/Fox play:    {model_unseen / n:>7.2%}")
    print(f"  the human frames themselves (sanity check):        {human_unseen / n:>7.2%}")

    print("\n  examples of what the model invented:")
    shown = 0
    for row in sampled_idx:
        if tuple(row) in support or shown >= 6:
            continue
        b, m, c, t = row
        print(f"      {describe_buttons(int(b)):<14} {_label('main_stick', int(m), centers):<22}"
              f"{_label('c_stick', int(c), centers):<24}{_label('triggers', int(t), centers)}")
        shown += 1

    # The marginals above are near-deterministic. Measure that directly, and measure
    # what it implies for the only thing decode does: deciding whether to change input.
    print("\n" + "=" * 78)
    print("HOW OFTEN DOES THE MODEL DECIDE TO CHANGE ITS INPUT?")
    print("=" * 78)
    print(f"\n{'group':<14}{'mean entropy':>14}   |{'model samples':>15}{'human':>9}{'ratio':>8}")
    print(f"{'':<14}{'(bits)':>14}   |{'a change':>15}{'changes':>9}")
    for g_i, g in enumerate(exp._GROUP_NAMES):
        p = probs[g].reshape(-1, probs[g].shape[-1])
        entropy = float(-(p * p.clamp_min(1e-12).log2()).sum(-1).mean())
        model_change = float((model_next[:, g_i] != held[:, g_i]).mean())
        human_change = float((human_next[:, g_i] != held[:, g_i]).mean())
        print(f"{g:<14}{entropy:>14.3f}   |{model_change:>15.2%}{human_change:>9.2%}"
              f"{model_change / max(human_change, 1e-9):>7.2f}x")

    model_any = float((model_next != held).any(axis=1).mean())
    human_any = float((human_next != held).any(axis=1).mean())
    print(f"\n  any group changes:  model {model_any:.2%}   human {human_any:.2%}"
          f"   ({model_any / max(human_any, 1e-9):.2f}x)")

    # Entropy is the model's own uncertainty. NLL is its cost against reality. Scored on
    # the SAME frames, the gap between them is calibration: how honest its confidence is.
    print("\n" + "=" * 78)
    print("ENTROPY vs NLL: is the model's confidence honest?")
    print("=" * 78)
    print(f"\n{'group':<14}{'entropy':>10}{'NLL':>10}{'gap':>10}{'effective':>12}   verdict")
    print(f"{'':<14}{'(bits)':>10}{'(bits)':>10}{'':>10}{'choices':>12}")
    for g_i, g in enumerate(exp._GROUP_NAMES):
        p = probs[g].reshape(B, L, -1)[:, :-1].reshape(-1, probs[g].shape[-1])
        entropy = float(-(p * p.clamp_min(1e-12).log2()).sum(-1).mean())
        target = torch.from_numpy(human_next[:, g_i]).to(p.device)
        nll = float(-p.gather(1, target[:, None]).clamp_min(1e-12).log2().mean())
        gap = nll - entropy
        verdict = "overconfident" if gap > 0.05 else ("underconfident" if gap < -0.05 else "well calibrated")
        print(f"{g:<14}{entropy:>10.3f}{nll:>10.3f}{gap:>+10.3f}{2 ** entropy:>12.2f}   {verdict}")


if __name__ == "__main__":
    main()
