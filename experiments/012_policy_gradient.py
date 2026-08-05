"""On-policy policy-gradient fine-tune for the GPT action policy.

This is intentionally small: start from a supervised 009 checkpoint, collect
closed-loop Fox-vs-Fox rollouts against an in-game CPU, and update the logprob
of the sampled actions with a final stock/damage reward. A small supervised
behavior-cloning term can be mixed in to keep the policy near the replay model.

Run on Vast, for example:

    uv run experiments/012_policy_gradient.py \
      --base-run 260615-041800_gpt-d256-L8-h4-Lc256_ranked-anon-1_fox-vs-fox \
      --base-ckpt final.pt \
      --rollout-iters 20
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import Any

import melee
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from loguru import logger
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal import streams
from hal.data.stats import FeatureStats
from hal.eval.harness import default_session_cfg
from hal.eval.harness import run_matches_vec
from hal.eval.scoring import summarize_trajectory
from hal.sim.inputs import ControllerInputs
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.trajectory import Trajectory
from hal.sim.vec import Slot
from hal.sim.vec import VecMatch
from hal.training.canonical import flatten_canonical_frame
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import download_latest
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import _live_batch_from_rolling
from hal.training.dataloader import make_loader
from hal.training.features import Context
from hal.training.features import action_vec_to_controller
from hal.training.features import preprocess
from hal.training.runs import make_run_name
from hal.training.runs import setup_run_dir
from hal.training.stats import load_consolidated_stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_EXP009_PATH = _ROOT / "experiments" / "009_simplify_arch.py"
_EXP009_SPEC = importlib.util.spec_from_file_location("hal_exp009_simplify_arch", _EXP009_PATH)
if _EXP009_SPEC is None or _EXP009_SPEC.loader is None:
    raise ImportError(f"could not load {_EXP009_PATH}")
exp009 = importlib.util.module_from_spec(_EXP009_SPEC)
sys.modules[_EXP009_SPEC.name] = exp009
_EXP009_SPEC.loader.exec_module(exp009)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, slots=True)
class ActionRecord:
    match: int
    features: dict[str, Tensor]
    ctx_pad: int
    action_idx: Tensor


@dataclass
class _SlotState:
    flat_hist: list[dict] = field(default_factory=list)
    ego_inputs_hist: list[np.ndarray] = field(default_factory=list)


def _logprob_and_entropy(model: Any, ctx: Context, action_idx: Tensor, *, temp: float) -> tuple[Tensor, Tensor]:
    logits = model(ctx.features, ctx.ctx_pad)[:, -1]
    logprob = torch.zeros(logits.shape[0], device=logits.device)
    entropy = torch.zeros_like(logprob)
    for g in range(exp009.N_GROUPS):
        lo = exp009._GROUP_OFFSETS[g]
        lg = logits[:, lo : lo + exp009._GROUP_VOCABS[g]] / temp
        logp = F.log_softmax(lg, dim=-1)
        probs = logp.exp()
        logprob = logprob + logp.gather(1, action_idx[:, g : g + 1]).squeeze(1)
        entropy = entropy - (probs * logp).sum(dim=-1)
    return logprob, entropy


def _sample_action(model: Any, ctx: Context, *, temp: float) -> tuple[Tensor, Tensor]:
    logits = model(ctx.features, ctx.ctx_pad)[:, -1]
    picks: list[Tensor] = []
    for g in range(exp009.N_GROUPS):
        lo = exp009._GROUP_OFFSETS[g]
        lg = logits[:, lo : lo + exp009._GROUP_VOCABS[g]] / temp
        picks.append(torch.multinomial(F.softmax(lg, dim=-1), 1).squeeze(-1))
    idx = torch.stack(picks, dim=-1)
    action = exp009._dequantize(model, idx)
    return idx, action


class PolicyGradientPolicy:
    """BatchPolicy that samples one 009 action per frame and records train samples."""

    def __init__(
        self,
        model: Any,
        stats: dict[str, FeatureStats],
        cfg: Any,
        *,
        temp: float,
        record_every: int,
        device: str = DEVICE,
    ) -> None:
        self.model = model
        self.stats = stats
        self.cfg = cfg
        self.temp = temp
        self.record_every = record_every
        self.device = device
        self.records: list[ActionRecord] = []
        self._slots: dict[Slot, _SlotState] = {}

    def __call__(self, frame_index: int, obs: dict[Slot, dict]) -> dict[Slot, ControllerInputs]:
        live = list(obs)
        for slot in live:
            st = self._slots.setdefault(slot, _SlotState())
            st.flat_hist.append(flatten_canonical_frame(obs[slot]))
            if len(st.flat_hist) > self.cfg.L_ctx:
                st.flat_hist.pop(0)

        ctx = self._context(live)
        with torch.no_grad():
            idx, actions = _sample_action(self.model, ctx, temp=self.temp)
        actions_np = actions.detach().cpu().numpy()

        if self.record_every > 0 and frame_index % self.record_every == 0:
            for i, slot in enumerate(live):
                self.records.append(
                    ActionRecord(
                        match=slot.match,
                        features={k: v[i].detach().cpu().clone() for k, v in ctx.features.items()},
                        ctx_pad=int(ctx.ctx_pad[i].item()),
                        action_idx=idx[i].detach().cpu().clone(),
                    )
                )

        out: dict[Slot, ControllerInputs] = {}
        for slot, action in zip(live, actions_np, strict=True):
            st = self._slots[slot]
            st.ego_inputs_hist.append(action.astype(np.float32))
            if len(st.ego_inputs_hist) > self.cfg.L_ctx:
                st.ego_inputs_hist.pop(0)
            out[slot] = action_vec_to_controller(action)
        return out

    def _context(self, live: list[Slot]) -> Context:
        per_slot = [
            _live_batch_from_rolling(
                self._slots[slot].flat_hist,
                self._slots[slot].ego_inputs_hist,
                ego_prefix="p1" if slot.port == 1 else "p2",
                L_ctx=self.cfg.L_ctx,
            )
            for slot in live
        ]
        stacked = {k: np.concatenate([d[k] for d in per_slot], axis=0) for k in per_slot[0]}
        feats = {k: v.to(self.device) for k, v in preprocess(stacked, self.stats).items()}
        ctx_pad = torch.tensor(
            [max(0, self.cfg.L_ctx - len(self._slots[slot].flat_hist)) for slot in live],
            dtype=torch.long,
            device=self.device,
        )
        return Context(features=feats, ctx_pad=ctx_pad)


def _collate_records(records: list[ActionRecord], *, device: str) -> tuple[Context, Tensor]:
    keys = sorted(set().union(*(r.features.keys() for r in records)))
    features: dict[str, Tensor] = {}
    for key in keys:
        example = next(r.features[key] for r in records if key in r.features)
        zero = torch.zeros_like(example)
        features[key] = torch.stack([r.features.get(key, zero) for r in records]).to(device)
    ctx_pad = torch.tensor([r.ctx_pad for r in records], dtype=torch.long, device=device)
    action_idx = torch.stack([r.action_idx for r in records]).long().to(device)
    return Context(features=features, ctx_pad=ctx_pad), action_idx


def _reward(traj: Trajectory | None, *, stock_weight: float, damage_weight: float, crash_penalty: float) -> float:
    if traj is None:
        return crash_penalty
    s = summarize_trajectory(traj)
    stock_diff = s.p1_stocks_left - s.p2_stocks_left
    damage_diff = (s.p2_damage_taken - s.p1_damage_taken) / 100.0
    return stock_weight * stock_diff + damage_weight * damage_diff


def _make_matches(n: int, *, stage: melee.Stage, cpu_level: int) -> list[VecMatch]:
    return [
        VecMatch(
            matchup=Matchup(
                stage=stage,
                players=(
                    PlayerSetup(port=1, character=melee.Character.FOX, cpu_level=0),
                    PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=cpu_level),
                ),
            ),
            model_ports=(1,),
        )
        for _ in range(n)
    ]


def _load_base(args: Args) -> tuple[Any, Any, dict[str, FeatureStats], dict]:
    if args.ckpt is not None:
        ckpt = Path(args.ckpt)
    else:
        ckpt = download_latest(args.base_run, Path("runs") / args.base_run, name=args.base_ckpt)
        if ckpt is None:
            raise SystemExit(f"no R2 checkpoint runs/{args.base_run}/{args.base_ckpt}")
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    cfg = exp009.TrainConfig(**state["cfg"])
    model = exp009.GPT(cfg).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    logger.info(f"loaded base checkpoint {ckpt} step={state.get('step')} device={DEVICE}")
    return model, cfg, stats, state


def _next_bc_batch(iterator: Iterable, loader: Iterable) -> tuple[Any, Iterable]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _bc_loss(model: Any, batch: Any) -> Tensor:
    comps = exp009.action_loss(model, batch)
    return sum(comps.values()).mean()


def _update_policy(
    model: Any,
    opt: torch.optim.Optimizer,
    records: list[ActionRecord],
    rewards_by_match: dict[int, float],
    *,
    temp: float,
    minibatch_size: int,
    update_epochs: int,
    entropy_weight: float,
    bc_weight: float,
    bc_loader: Iterable | None,
    bc_iterator: Iterable | None,
) -> tuple[dict[str, float], Iterable | None]:
    if not records:
        return {"loss": math.nan, "pg_loss": math.nan, "entropy": math.nan, "bc_loss": math.nan}, bc_iterator

    rewards = torch.tensor([rewards_by_match[r.match] for r in records], dtype=torch.float32)
    adv = rewards - rewards.mean()
    if adv.std(unbiased=False) > 1e-6:
        adv = adv / adv.std(unbiased=False)

    losses: list[float] = []
    pg_losses: list[float] = []
    entropies: list[float] = []
    bc_losses: list[float] = []
    model.train()
    for _ in range(update_epochs):
        order = torch.randperm(len(records))
        for start in range(0, len(records), minibatch_size):
            idx = order[start : start + minibatch_size]
            batch_records = [records[int(i)] for i in idx]
            ctx, action_idx = _collate_records(batch_records, device=DEVICE)
            adv_batch = adv[idx].to(DEVICE)

            logprob, entropy = _logprob_and_entropy(model, ctx, action_idx, temp=temp)
            pg_loss = -(adv_batch * logprob).mean()
            loss = pg_loss - entropy_weight * entropy.mean()
            bc_value = torch.tensor(0.0, device=DEVICE)
            if bc_weight > 0 and bc_loader is not None:
                assert bc_iterator is not None
                bc_batch, bc_iterator = _next_bc_batch(bc_iterator, bc_loader)
                bc_value = _bc_loss(model, bc_batch.to(DEVICE))
                loss = loss + bc_weight * bc_value

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))
            pg_losses.append(float(pg_loss.detach().cpu()))
            entropies.append(float(entropy.mean().detach().cpu()))
            bc_losses.append(float(bc_value.detach().cpu()))
    model.eval()
    return (
        {
            "loss": float(np.mean(losses)),
            "pg_loss": float(np.mean(pg_losses)),
            "entropy": float(np.mean(entropies)),
            "bc_loss": float(np.mean(bc_losses)),
        },
        bc_iterator,
    )


@dataclass
class Args:
    base_run: str = "260615-041800_gpt-d256-L8-h4-Lc256_ranked-anon-1_fox-vs-fox"
    """Run name under R2 runs/ to initialize from."""
    base_ckpt: str = "final.pt"
    """Checkpoint object under runs/<base-run>/ to initialize from."""
    ckpt: str | None = None
    """Local checkpoint path. Overrides base_run/base_ckpt."""
    rollout_iters: int = 20
    rollout_replicas: int = 4
    max_frames: int = 7200
    record_every: int = 8
    update_epochs: int = 1
    minibatch_size: int = 256
    lr: float = 1e-5
    weight_decay: float = 0.0
    temp: float | None = None
    entropy_weight: float = 1e-3
    bc_weight: float = 0.01
    bc_batch_size: int = 64
    bc_cache_limit_gb: int = 40
    bc_num_workers: int = 2
    stock_weight: float = 1.0
    damage_weight: float = 1.0
    crash_penalty: float = -2.0
    cpu_level: int = 9
    stage: melee.Stage = melee.Stage.FINAL_DESTINATION
    save_every: int = 5
    seed: int = 0
    comment: str = "fox-vs-fox-cpu-rl"


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    model, cfg, stats, base_state = _load_base(args)
    temp = cfg.decode_temp if args.temp is None else args.temp
    run_name = make_run_name(
        f"pg-gpt-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}",
        cfg.data_root,
        args.comment,
    )
    ckpt_dir, replay_dir = setup_run_dir(run_name)
    uploader = BackgroundUploader(run_name)
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    sched = LambdaLR(opt, lambda _: 1.0)

    bc_loader = None
    bc_iterator = None
    if args.bc_weight > 0:
        bc_loader = make_loader(
            split="train",
            data_root=cfg.data_root,
            remote=streams.remote_for_local(cfg.data_root),
            cache_limit=f"{args.bc_cache_limit_gb}gb",
            shuffle_block_size=cfg.shuffle_block_size,
            stats=stats,
            L_ctx=cfg.L_ctx,
            L_chunk=exp009.L_CHUNK,
            batch_size=args.bc_batch_size,
            seed=args.seed,
            character_pair=cfg.character_pair,
            num_workers=args.bc_num_workers,
            prefetch_factor=2,
        )
        bc_iterator = iter(bc_loader)

    rl_config = asdict(args)
    rl_config["stage"] = args.stage.name
    wandb.init(
        project="hal",
        name=run_name,
        tags=["rl", "policy_gradient", "gpt", "fox_vs_fox"],
        config={
            "rl": rl_config,
            "base_step": base_state.get("step"),
            "base_cfg": asdict(cfg),
        },
    )
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    logger.info(f"run={run_name} temp={temp} rollout_replicas={args.rollout_replicas}")

    session_cfg = default_session_cfg(replay_dir)
    for it in range(args.rollout_iters):
        t0 = time.monotonic()
        policies: list[PolicyGradientPolicy] = []

        def policy_factory(policies: list[PolicyGradientPolicy] = policies) -> PolicyGradientPolicy:
            policy = PolicyGradientPolicy(
                model,
                stats,
                cfg,
                temp=temp,
                record_every=args.record_every,
                device=DEVICE,
            )
            policies.append(policy)
            return policy

        iter_replay_dir = replay_dir / f"iter_{it:04d}"
        session_cfg = replace(session_cfg, replay_dir=str(iter_replay_dir))
        matches = _make_matches(args.rollout_replicas, stage=args.stage, cpu_level=args.cpu_level)
        trajs = run_matches_vec(
            session_cfg,
            matches,
            policy_factory,
            max_frames=args.max_frames,
            max_parallel=args.rollout_replicas,
            start_retries=0,
        )
        records = list(itertools.chain.from_iterable(p.records for p in policies))
        rewards_by_match = {
            i: _reward(
                traj,
                stock_weight=args.stock_weight,
                damage_weight=args.damage_weight,
                crash_penalty=args.crash_penalty,
            )
            for i, traj in enumerate(trajs)
        }
        update, bc_iterator = _update_policy(
            model,
            opt,
            records,
            rewards_by_match,
            temp=temp,
            minibatch_size=args.minibatch_size,
            update_epochs=args.update_epochs,
            entropy_weight=args.entropy_weight,
            bc_weight=args.bc_weight,
            bc_loader=bc_loader,
            bc_iterator=bc_iterator,
        )
        sched.step()

        summaries = [summarize_trajectory(t) for t in trajs if t is not None]
        reward_values = list(rewards_by_match.values())
        log = {
            "global_step": it,
            "rl/reward_mean": float(np.mean(reward_values)),
            "rl/reward_min": float(np.min(reward_values)),
            "rl/reward_max": float(np.max(reward_values)),
            "rl/records": len(records),
            "rl/rollout_s": time.monotonic() - t0,
            "rl/successful_matches": len(summaries),
            "rl/p1_stocks_left": float(np.mean([s.p1_stocks_left for s in summaries])) if summaries else 0.0,
            "rl/p2_stocks_left": float(np.mean([s.p2_stocks_left for s in summaries])) if summaries else 0.0,
            "rl/p1_damage_taken": float(np.mean([s.p1_damage_taken for s in summaries])) if summaries else 0.0,
            "rl/p2_damage_taken": float(np.mean([s.p2_damage_taken for s in summaries])) if summaries else 0.0,
            "train/lr": opt.param_groups[0]["lr"],
            **{f"train/{k}": v for k, v in update.items()},
        }
        wandb.log(log)
        logger.info(
            f"iter {it}: reward={log['rl/reward_mean']:.3f} records={len(records)} "
            f"loss={update['loss']:.3f} rollout_s={log['rl/rollout_s']:.1f}"
        )
        if args.save_every > 0 and (it + 1) % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / "latest.pt",
                step=it,
                model=model,
                opt=opt,
                sched=sched,
                cfg=asdict(cfg),
                wandb_id=wandb.run.id if wandb.run is not None else None,
                uploader=uploader,
            )
    save_checkpoint(
        ckpt_dir / "final.pt",
        step=args.rollout_iters,
        model=model,
        opt=opt,
        sched=sched,
        cfg=asdict(cfg),
        wandb_id=wandb.run.id if wandb.run is not None else None,
        uploader=uploader,
    )
    uploader.upload_tree(replay_dir, base=ckpt_dir, pattern="*.slp")
    uploader.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
