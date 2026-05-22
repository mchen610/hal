import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import math
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import WindowSampler
from prepare import collate_windows
from prepare import make_loader as _make_loader
from prepare import relabel_ego as _relabel_ego
from streaming import StreamingDataLoader
from streaming import StreamingDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal.data.stats import FeatureStats
from hal.data.stats import FeatureStatsSufficient
from hal.data.stats import load_sufficient_stats
from hal.data.stats import merge_sufficient
from hal.sim.inputs import ControllerInputsValue
from hal.wire import BUTTON_BITS
from hal.wire import mask_value

torch.set_printoptions(linewidth=300)

# %%
DATA_ROOT = "data/processed/ranked-anonymized-1/mds"
L_CTX = 256
L_CHUNK = 16
A_DIM = 15  # 4 sticks + 2 triggers + 9 buttons
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = dict(
    d_model=256,
    n_layers=6,
    n_heads=8,
    dim_feedforward=1024,
    dropout=0.1,
    time_emb_dim=128,
    L_ctx=L_CTX,
    L_chunk=L_CHUNK,
    n_flow_steps=8,
    batch_size=32,
    lr=3e-4,
    weight_decay=0.01,
    warmup_steps=500,
    max_steps=15_000,
    val_every=500,
    val_n_batches=16,
    eval_every=2500,
    eval_max_frames=3600,
    num_workers=8,
    prefetch_factor=8,
    data_root=DATA_ROOT,
    # P(zero ego history's [:k] for a random k ~ U{0..L_ctx}) per sample at train.
    # Trains the model to handle the closed-loop rolling-buffer transient where
    # _ego_inputs_hist starts as L_ctx zeros and slowly fills with model outputs.
    ego_history_dropout_prob=0.5,
    # Latency-aware (receding-horizon) chunked prediction. K=4 frames @ 60Hz =
    # 66ms ~ one inference period; the model conditions on K already-committed
    # bridge actions [t, t+K) and predicts L_chunk frames starting at t+K.
    # K=0 reproduces the original open-loop architecture.
    latency_frames=4,
)


# %%
def _consolidate_key(name: str) -> str:
    """Strip port prefix (`p1_`, `p2_`, `ego_`, `opp_`) so symmetric features share a stats entry."""
    for pre in ("p1_", "p2_", "ego_", "opp_"):
        if name.startswith(pre):
            return name[len(pre) :]
    return name


def load_consolidated_stats(path: Path) -> dict[str, FeatureStats]:
    """Welford-merge sufficient stats across p1/p2 ports before finalizing."""
    merged: dict[str, FeatureStatsSufficient] = {}
    for name, block in load_sufficient_stats(path).items():
        key = _consolidate_key(name)
        merged[key] = merge_sufficient(merged[key], block) if key in merged else block
    return {k: b.finalize() for k, b in merged.items()}


stats = load_consolidated_stats(Path(DATA_ROOT) / "stats.json")

# %%
# Feature schema for the model. The classifier below routes every MDS column
# into exactly one of these buckets (or drops it).
FLOAT_FEATURES: tuple[str, ...] = (
    "position_x",
    "position_y",
    "percent",
    "shield",
    "direction",
    "hitlag_left",
)

# (vocab_size, embed_dim) per categorical feature. action vocab rounded up
# from libmelee's ~395 to a safe power-of-two-ish.
CAT_FEATURES: dict[str, tuple[int, int]] = {
    "action": (512, 32),
    "stock": (5, 8),
    "jumps_used": (9, 8),
    "hurtbox_state": (4, 8),
    "airborne": (2, 4),
}

# Canonical ordering of the 15-channel ego action vector. Matches
# ControllerInputsValue field order for sticks/triggers and BUTTON_BITS for
# buttons. Used both as model target and to construct ControllerInputsValue
# at inference — must stay in lockstep with action_vec_to_controller.
ACTION_CHANNELS: tuple[str, ...] = (
    "main_stick_x",
    "main_stick_y",
    "c_stick_x",
    "c_stick_y",
    "trigger_l_physical",
    "trigger_r_physical",
    "button_a",
    "button_b",
    "button_x",
    "button_y",
    "button_z",
    "button_r",
    "button_l",
    "button_start",
    "button_d_up",
)
assert len(ACTION_CHANNELS) == A_DIM

_BUTTON_ORDER = ("a", "b", "x", "y", "z", "r", "l", "start", "d_up")

# Patterns dropped before any feature work. raw_* is dropped because user
# chose logical sticks; nana_* is skipped for the toy; trigger_logical is
# redundant with the physical l/r channels we already consume.
_DROP_PATTERNS = ("_raw_", "_nana_", "_trigger_logical")


def _classify(name: str) -> str:
    if name == "frame":
        return "drop"
    if any(p in name for p in _DROP_PATTERNS):
        return "drop"
    if name.endswith("_action_frame"):
        return "action_frame"
    if any(name.endswith(f"_{c}") for c in CAT_FEATURES):
        return "cat"
    if "_button_" in name:
        return "button"
    if any(
        name.endswith(f"_{s}")
        for s in (
            "main_stick_x",
            "main_stick_y",
            "c_stick_x",
            "c_stick_y",
            "trigger_l_physical",
            "trigger_r_physical",
        )
    ):
        return "stick_trigger"
    if any(name.endswith(f"_{f}") for f in FLOAT_FEATURES):
        return "float"
    return "drop"


def _is_masked(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "f":
        return np.isnan(arr)
    return arr == mask_value(arr.dtype)


def _normalize(arr: np.ndarray, s: FeatureStats) -> np.ndarray:
    return (2.0 * (arr - s.min) / (s.max - s.min) - 1.0).astype(np.float32)


def _standardize(arr: np.ndarray, s: FeatureStats) -> np.ndarray:
    return ((arr - s.mean) / s.std).astype(np.float32)


def preprocess_inputs(
    batch: dict[str, np.ndarray],
    feature_stats: dict[str, FeatureStats],
) -> dict[str, torch.Tensor]:
    """Tokenizer-style: per-feature sanitization + per-float mask sidecars.
    Operates on either single-sample [L] arrays or batched [B, L]; the numpy
    ops broadcast either way and torch.from_numpy preserves the shape.
    """
    out: dict[str, torch.Tensor] = {}
    for name, arr in batch.items():
        kind = _classify(name)
        if kind == "drop":
            continue
        mask = _is_masked(arr)
        if kind == "button" or kind == "stick_trigger":
            x = np.where(mask, 0.0, arr).astype(np.float32)
        elif kind == "cat":
            x = np.where(mask, 0, arr).astype(np.int64)
        elif kind == "action_frame":
            # Vocab unbounded — treat as a float. /60 puts most values in [0, ~5].
            x = np.where(mask, 0, arr).astype(np.float32) / 60.0
        elif kind == "float":
            s = feature_stats[_consolidate_key(name)]
            x = _standardize(arr, s) if "position" in name else _normalize(arr, s)
            x = np.where(mask, 0.0, x)
        else:
            raise AssertionError(f"unhandled kind {kind} for {name}")
        out[name] = torch.from_numpy(np.ascontiguousarray(x))
        if kind == "float" and mask.any():
            out[f"{name}_mask"] = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))
    return out


# %%
# WindowSampler / collate_windows / relabel_ego live in `prepare.py` so they
# can be re-imported by DataLoader worker processes without re-running these
# cells. See prepare.py.


# %%
def stack_ego_actions(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """[B, L, A_DIM] — stack ego action channels in canonical order."""
    return torch.stack([batch[f"ego_{ch}"] for ch in ACTION_CHANNELS], dim=-1)


def action_vec_to_controller(a: np.ndarray) -> ControllerInputsValue:
    """One 15-vector → one ControllerInputsValue. Buttons threshold at 0.5."""
    a = np.asarray(a).reshape(-1)
    buttons = 0
    for i, name in enumerate(_BUTTON_ORDER):
        if a[6 + i] > 0.5:
            buttons |= BUTTON_BITS[name]
    return ControllerInputsValue(
        main_x=float(np.clip(a[0], -1.0, 1.0)),
        main_y=float(np.clip(a[1], -1.0, 1.0)),
        c_x=float(np.clip(a[2], -1.0, 1.0)),
        c_y=float(np.clip(a[3], -1.0, 1.0)),
        trigger_l=float(np.clip(a[4], 0.0, 1.0)),
        trigger_r=float(np.clip(a[5], 0.0, 1.0)),
        buttons=int(buttons),
    )


_NEUTRAL_ACTION = np.zeros(A_DIM, dtype=np.float32)


# %%
def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: [B] in [0, 1] → [B, dim]."""
    half = dim // 2
    device = t.device
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / half)
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowMatchingPolicy(nn.Module):
    """Unified Transformer over [L_ctx ctx tokens | K bridge tokens | L_chunk noise tokens].
    Context tokens carry observed ego+opp gamestate + ego controller history.
    Bridge tokens (when ``latency_frames`` K > 0) carry the K already-committed
    actions about to execute while the new chunk is being computed at 60ms /
    15 Hz cadence; they get their own type embedding and a separate projection.
    Chunk tokens carry the noised action a_t + time embedding + a learned
    chunk-type embedding. Output head reads the chunk positions and predicts
    the flow-matching velocity v̂ ∈ R^{L_chunk × A_DIM}.

    K=0 reproduces the original open-loop architecture exactly (bridge_proj /
    bridge_type_emb / extended pos_emb are not created), keeping older
    checkpoints loadable.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.L_ctx = cfg["L_ctx"]
        self.L_chunk = cfg["L_chunk"]
        self.K = int(cfg.get("latency_frames", 0))
        d = cfg["d_model"]
        self.time_emb_dim = cfg["time_emb_dim"]
        self.ego_history_dropout_prob = float(cfg.get("ego_history_dropout_prob", 0.0))

        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )

        n_float = len(FLOAT_FEATURES)
        n_mask = len(FLOAT_FEATURES)
        n_action_frame = 1
        n_cat = sum(dim for _, dim in CAT_FEATURES.values())
        per_player_dim = n_float + n_mask + n_action_frame + n_cat
        per_frame_in_dim = 2 * per_player_dim + A_DIM  # ego + opp + ego controller history

        self.per_frame_in_dim = per_frame_in_dim
        self.ctx_proj = nn.Linear(per_frame_in_dim, d)
        self.chunk_proj = nn.Linear(A_DIM, d)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.chunk_type_emb = nn.Parameter(torch.zeros(d))
        if self.K > 0:
            self.bridge_proj = nn.Linear(A_DIM, d)
            self.bridge_type_emb = nn.Parameter(torch.zeros(d))
        self.pos_emb = nn.Embedding(self.L_ctx + self.K + self.L_chunk, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["n_heads"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg["n_layers"])
        self.head = nn.Linear(d, A_DIM)

    def _per_player_features(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        """[B, L_ctx, per_player_dim] — all observed features for one player."""
        L_ctx = self.L_ctx
        ref = batch[f"{prefix}_position_x"]
        B = ref.shape[0]
        device = ref.device
        parts: list[torch.Tensor] = []
        for f in FLOAT_FEATURES:
            parts.append(batch[f"{prefix}_{f}"][:, :L_ctx, None])
        for f in FLOAT_FEATURES:
            mk = f"{prefix}_{f}_mask"
            if mk in batch:
                parts.append(batch[mk][:, :L_ctx, None])
            else:
                parts.append(torch.zeros(B, L_ctx, 1, device=device))
        parts.append(batch[f"{prefix}_action_frame"][:, :L_ctx, None])
        for cat_name, (vocab, _) in CAT_FEATURES.items():
            ids = batch[f"{prefix}_{cat_name}"][:, :L_ctx].clamp(0, vocab - 1)
            parts.append(self.cat_embeds[cat_name](ids))
        return torch.cat(parts, dim=-1)

    def _ego_history_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """[B, L_ctx, A_DIM] — ego's real past controller inputs.

        During training, with probability ``ego_history_dropout_prob`` per
        sample, zero a random *left* prefix ``[:k]`` (k ~ U{0..L_ctx}, inclusive
        of L_ctx so full-history zero is reachable). Matches the closed-loop
        rolling-buffer transient where ``_ego_inputs_hist`` starts as L_ctx
        zeros and is gradually displaced by live predictions.
        """
        L_ctx = self.L_ctx
        hist = torch.cat([batch[f"ego_{ch}"][:, :L_ctx, None] for ch in ACTION_CHANNELS], dim=-1)
        if self.training and self.ego_history_dropout_prob > 0:
            B = hist.size(0)
            device = hist.device
            apply = torch.rand(B, device=device) < self.ego_history_dropout_prob
            ks = torch.randint(0, L_ctx + 1, (B,), device=device)
            ks = torch.where(apply, ks, torch.zeros_like(ks))
            positions = torch.arange(L_ctx, device=device)[None, :].expand(B, L_ctx)
            mask = positions < ks[:, None]  # [B, L_ctx]
            hist = hist.masked_fill(mask[..., None], 0.0)
        return hist

    def build_context_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ego = self._per_player_features(batch, "ego")
        opp = self._per_player_features(batch, "opp")
        hist = self._ego_history_features(batch)
        return self.ctx_proj(torch.cat([ego, opp, hist], dim=-1))

    def forward(
        self,
        batch_or_ctx: dict[str, torch.Tensor] | torch.Tensor,
        a_t: torch.Tensor,
        t: torch.Tensor,
        bridge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        batch_or_ctx — either a raw preprocessed batch dict OR a precomputed
                       [B, L_ctx, d] context token tensor (useful when
                       integrating the flow at inference, where context is
                       fixed across all flow steps).
        a_t          — [B, L_chunk, A_DIM] noised action chunk.
        t            — [B] or any shape that flattens to [B] of times in [0, 1].
        bridge       — [B, K, A_DIM] clean actions for frames [t, t+K) that
                       will execute while this prediction is being computed.
                       Required iff self.K > 0; ignored otherwise.
        Returns v_pred [B, L_chunk, A_DIM].
        """
        if isinstance(batch_or_ctx, dict):
            ctx_tokens = self.build_context_tokens(batch_or_ctx)
        else:
            ctx_tokens = batch_or_ctx
        chunk_tokens = self.chunk_proj(a_t)
        t_flat = t.reshape(-1)
        t_emb = sinusoidal_time_embedding(t_flat, self.time_emb_dim)
        t_proj = self.time_mlp(t_emb)
        chunk_tokens = chunk_tokens + t_proj[:, None, :] + self.chunk_type_emb[None, None, :]
        if self.K > 0:
            if bridge is None:
                raise ValueError(f"latency_frames={self.K} requires bridge tensor of shape [B, {self.K}, A_DIM]")
            if bridge.shape[1] != self.K:
                raise ValueError(f"bridge length {bridge.shape[1]} != K={self.K}")
            bridge_tokens = self.bridge_proj(bridge) + self.bridge_type_emb[None, None, :]
            seq = torch.cat([ctx_tokens, bridge_tokens, chunk_tokens], dim=1)
        else:
            seq = torch.cat([ctx_tokens, chunk_tokens], dim=1)
        pos_ids = torch.arange(seq.size(1), device=seq.device)
        seq = seq + self.pos_emb(pos_ids)[None, :, :]
        out = self.encoder(seq)
        chunk_start = self.L_ctx + self.K
        return self.head(out[:, chunk_start:, :])


# %%
def make_run_name(cfg: dict, comment: str = "") -> str:
    """`YYMMDD-HHMMSS_fm-d256-L6-H8-Lc256-Lk16-fs8_ranked-anon-1[_comment]`."""
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    model_tag = (
        f"fm-d{cfg['d_model']}-L{cfg['n_layers']}-H{cfg['n_heads']}"
        f"-Lc{cfg['L_ctx']}-Lk{cfg['L_chunk']}-fs{cfg['n_flow_steps']}"
    )
    data_tag = Path(cfg["data_root"]).parent.name.replace("anonymized", "anon")
    parts = [stamp, model_tag, data_tag]
    if comment:
        parts.append(comment)
    return "_".join(parts)


# %%
def _to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_loader(cfg: dict, split: str) -> StreamingDataLoader:
    """Thin wrapper around prepare.make_loader that unpacks cfg."""
    return _make_loader(
        data_root=cfg["data_root"],
        split=split,
        L_ctx=cfg["L_ctx"],
        L_chunk=cfg["L_chunk"],
        K=int(cfg.get("latency_frames", 0)),
        batch_size=cfg["batch_size"],
        num_workers=int(cfg.get("num_workers", 4)),
        prefetch_factor=int(cfg.get("prefetch_factor", 4)),
    )


@torch.no_grad()
def build_val_cache(val_loader: StreamingDataLoader, n_batches: int, cfg: dict, device: str) -> list[tuple]:
    """Materialize n_batches of val windows + fixed (t, z) noise on-device.
    Caching makes val loss comparable across evaluations — same windows,
    same noise — so a drop in val loss is a real model improvement, not
    sampling variance. CPU generator + .to(device) to avoid cuda-rng pinning."""
    L_ctx = cfg["L_ctx"]
    L_chunk = cfg["L_chunk"]
    K = int(cfg.get("latency_frames", 0))
    cache: list[tuple] = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for raw in val_loader:
        batch = _to_device(preprocess_inputs(raw, stats), device)
        actions_all = stack_ego_actions(batch)
        bridge = actions_all[:, L_ctx : L_ctx + K, :] if K > 0 else None
        a_target = actions_all[:, L_ctx + K :, :]
        B = a_target.shape[0]
        t = torch.rand(B, generator=g).to(device)
        z = torch.randn(B, L_chunk, A_DIM, generator=g).to(device)
        t_b = t.view(B, 1, 1)
        a_t = (1 - t_b) * z + t_b * a_target
        v_target = a_target - z
        cache.append((batch, a_t, t, v_target, bridge))
        if len(cache) >= n_batches:
            break
    if not cache:
        raise RuntimeError("val loader yielded zero batches")
    return cache


@torch.no_grad()
def val_loss(model: FlowMatchingPolicy, val_cache: list[tuple]) -> float:
    """Sample-weighted MSE across cached val batches. Toggles model.eval/train."""
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for batch, a_t, t, v_target, bridge in val_cache:
        v_pred = model(batch, a_t, t, bridge=bridge)
        total += F.mse_loss(v_pred, v_target).item() * v_target.shape[0]
        count += v_target.shape[0]
    if was_training:
        model.train()
    return total / count


def lr_schedule(cfg: dict):
    """Linear warmup → cosine to floor."""
    floor = 1e-5 / cfg["lr"]

    def fn(step: int) -> float:
        if step < cfg["warmup_steps"]:
            return step / max(1, cfg["warmup_steps"])
        progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
        progress = min(1.0, progress)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return floor + (1 - floor) * cos

    return fn


def train(
    model: FlowMatchingPolicy,
    loader: StreamingDataLoader,
    cfg: dict,
    device: str,
    val_loader: StreamingDataLoader,
    comment: str = "",
) -> None:
    import time

    run_name = make_run_name(cfg, comment)
    wandb.init(
        project="hal",
        name=run_name,
        tags=["toy", "flow-matching", f"d{cfg['d_model']}", f"L{cfg['n_layers']}"],
        config=cfg,
    )
    ckpt_dir = Path("runs") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] writing checkpoints to {ckpt_dir}", flush=True)

    def _save_ckpt(name: str, step: int) -> None:
        path = ckpt_dir / name
        torch.save({"step": step, "model": model.state_dict(), "cfg": cfg}, path)
        print(f"[ckpt] saved {path}", flush=True)

    print("[val] building cached val set…", flush=True)
    val_t0 = time.monotonic()
    val_cache = build_val_cache(val_loader, cfg["val_n_batches"], cfg, device)
    print(
        f"[val] cached {len(val_cache)} batches "
        f"({sum(b[3].shape[0] for b in val_cache)} samples) in {time.monotonic() - val_t0:.1f}s",
        flush=True,
    )

    opt = AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg["weight_decay"])
    sched = LambdaLR(opt, lr_schedule(cfg))
    model.train()
    print("[t+0.0s] building dataloader iter…", flush=True)
    it_t0 = time.monotonic()
    it = iter(loader)
    L_ctx = cfg["L_ctx"]
    K = int(cfg.get("latency_frames", 0))
    print(f"[t+{time.monotonic() - it_t0:.1f}s] iter built; fetching first batch…", flush=True)
    fetch_t0 = time.monotonic()
    raw = next(it)
    print(f"[t+{time.monotonic() - fetch_t0:.1f}s] first batch ready (B={cfg['batch_size']})", flush=True)
    have_first = True
    step_t0 = time.monotonic()
    run_t0 = time.monotonic()
    for step in range(cfg["max_steps"]):
        if not have_first:
            try:
                raw = next(it)
            except StopIteration:
                it = iter(loader)
                raw = next(it)
        have_first = False
        data_t0 = time.monotonic()
        batch = _to_device(preprocess_inputs(raw, stats), device)
        actions_all = stack_ego_actions(batch)
        bridge = actions_all[:, L_ctx : L_ctx + K, :] if K > 0 else None
        a_target = actions_all[:, L_ctx + K :, :]
        B = a_target.shape[0]
        t = torch.rand(B, device=device)
        z = torch.randn_like(a_target)
        t_b = t.view(B, 1, 1)
        a_t = (1 - t_b) * z + t_b * a_target
        v_target = a_target - z
        v_pred = model(batch, a_t, t, bridge=bridge)
        loss = F.mse_loss(v_pred, v_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step_dt = time.monotonic() - step_t0
        sps = B / step_dt
        wandb.log(
            {
                "train/loss": loss.item(),
                "train/lr": opt.param_groups[0]["lr"],
                "throughput/step_s": step_dt,
                "throughput/samples_per_s": sps,
                "throughput/data_overhead_s": data_t0 - step_t0,
            },
            step=step,
        )
        # Verbose for the first 20 steps so we see things move; then sparser.
        if step < 20 or step % 50 == 0:
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: loss {loss.item():.4f} "
                f"step_dt={step_dt * 1000:.0f}ms ({sps:.1f} samples/s)",
                flush=True,
            )
        step_t0 = time.monotonic()
        if cfg["val_every"] > 0 and step > 0 and step % cfg["val_every"] == 0:
            vl = val_loss(model, val_cache)
            wandb.log({"val/loss": vl}, step=step)
            print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: val_loss {vl:.4f}", flush=True)
        if cfg["eval_every"] > 0 and step > 0 and step % cfg["eval_every"] == 0:
            _save_ckpt(f"step_{step:06d}.pt", step)
            metrics = closed_loop_eval(model, max_frames=cfg["eval_max_frames"])
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: closed_loop {metrics}", flush=True)
    # Final ckpt + val + closed-loop eval so the last entry isn't blind.
    vl_final = val_loss(model, val_cache)
    wandb.log({"val/loss": vl_final}, step=cfg["max_steps"])
    print(f"[final] val_loss {vl_final:.4f}", flush=True)
    metrics_final = closed_loop_eval(model, max_frames=cfg["eval_max_frames"])
    wandb.log({f"eval/{k}": v for k, v in metrics_final.items()}, step=cfg["max_steps"])
    print(f"[final] closed_loop {metrics_final}", flush=True)
    _save_ckpt("final.pt", cfg["max_steps"])


# %%
def _flatten_canonical_frame(frame: dict) -> dict[str, float | int]:
    """Mirror hal/sim/trajectory.from_capture per-frame: nested canonical dict
    from Session.step() → flat MDS-shaped {p1_*, p2_*} gamestate dict.
    Only fills the gamestate fields the model consumes; controller history is
    stitched in separately by ModelControllerSource from inputs we punched."""
    out: dict[str, float | int] = {}
    for libmelee_port, prefix in ((1, "p1"), (2, "p2")):
        pd = frame["ports"].get(libmelee_port)
        if pd is None:
            continue
        post = pd["leader"]["post"]
        pos = post["position"]
        out[f"{prefix}_position_x"] = float(pos["x"])
        out[f"{prefix}_position_y"] = float(pos["y"])
        out[f"{prefix}_percent"] = float(post["percent"])
        out[f"{prefix}_shield"] = float(post["shield"])
        out[f"{prefix}_stock"] = int(post["stock"])
        out[f"{prefix}_direction"] = float(post["direction"])
        out[f"{prefix}_action"] = int(post["action"])
        # libmelee names it state_age; MDS calls the same field action_frame.
        out[f"{prefix}_action_frame"] = float(post.get("state_age") or 0.0)
        out[f"{prefix}_hitlag_left"] = float(post.get("hitlag_left") or 0.0)
        out[f"{prefix}_jumps_used"] = int(post.get("jumps_used") or 0)
        out[f"{prefix}_airborne"] = int(post.get("airborne") or 0)
        out[f"{prefix}_hurtbox_state"] = int(post.get("hurtbox_state") or 0)
    return out


def _live_batch_from_rolling(
    flat_history: list[dict],
    ego_inputs_hist: list[np.ndarray],
    ego_prefix: str,
) -> dict[str, np.ndarray]:
    """Build the [1, L_ctx] batch the model expects from rolling buffers."""
    out: dict[str, np.ndarray] = {}
    # Stack each gamestate column from the deque of flat frames.
    keys = flat_history[0].keys()
    for k in keys:
        sample = flat_history[0][k]
        dtype = np.int32 if isinstance(sample, int) else np.float32
        out[k] = np.array([h[k] for h in flat_history], dtype=dtype)
    # Inject ego controller history (our intended actions, not whatever
    # libmelee reads back). For buttons store as int 0/1 so classifier routes
    # them via "button".
    hist_arr = np.stack(ego_inputs_hist)  # [L_ctx, A_DIM]
    for i, ch in enumerate(ACTION_CHANNELS):
        col = hist_arr[:, i]
        if ch.startswith("button_"):
            out[f"{ego_prefix}_{ch}"] = (col > 0.5).astype(np.int32)
        else:
            out[f"{ego_prefix}_{ch}"] = col.astype(np.float32)
    # `frame` not needed by the model; drop to avoid confusing preprocess.
    out.pop("frame", None)
    relabeled = _relabel_ego(out, ego_prefix)
    return {k: v[None, ...] for k, v in relabeled.items()}


@torch.no_grad()
def _integrate_chunk_batched(
    model: FlowMatchingPolicy,
    batch: dict[str, torch.Tensor],
    n_steps: int,
    device: str,
    bridge: torch.Tensor | None = None,
) -> np.ndarray:
    """Euler-integrate from z ~ N(0,I) for n_steps. Returns [B, L_chunk, A_DIM].
    ``bridge`` is required iff ``model.K > 0``; it carries the K already-
    committed actions for frames [now, now+K) that execute during inference."""
    model.eval()
    ctx = model.build_context_tokens(batch)
    B = next(iter(batch.values())).shape[0]
    a = torch.randn(B, model.L_chunk, A_DIM, device=device)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t_val = torch.full((B,), k * dt, device=device)
        v = model(ctx, a, t_val, bridge=bridge)
        a = a + dt * v
    return a.cpu().numpy()


def _integrate_chunk(
    model: FlowMatchingPolicy,
    batch: dict[str, torch.Tensor],
    n_steps: int,
    device: str,
    bridge: torch.Tensor | None = None,
) -> np.ndarray:
    """Single-sample shim over _integrate_chunk_batched. Returns [L_chunk, A_DIM]."""
    return _integrate_chunk_batched(model, batch, n_steps, device, bridge=bridge)[0]


@dataclass
class ModelControllerSource:
    """ControllerSource: rolling-history flow-matching policy with optional
    receding-horizon (latency-aware) chunked prediction.

    K=0: open-loop — predict L_chunk frames, play them, replan after L_chunk
    frames.

    K>0: receding horizon — replan every K frames. Each chunk predicts
    L_chunk frames starting K ahead of the replan time (modeling inference
    latency). Between replans, play the K actions from the previous chunk's
    first K predictions (the "bridge"); at bootstrap (first chunk after
    warm-up), the bridge is zeros and the first K frames are played neutral.
    """

    model: nn.Module
    stats: dict
    ego_prefix: str
    L_ctx: int = L_CTX
    L_chunk: int = L_CHUNK
    K: int = 0
    n_flow_steps: int = 8
    device: str = DEVICE
    _flat_hist: list = field(default_factory=list)
    _ego_inputs_hist: list = field(default_factory=list)
    _pending: np.ndarray | None = None
    _current_bridge: np.ndarray | None = None
    _offset: int = 0

    def __call__(self, frame_index: int, last_gamestate: dict | None):
        if last_gamestate is not None:
            self._flat_hist.append(_flatten_canonical_frame(last_gamestate))
            if len(self._flat_hist) > self.L_ctx:
                self._flat_hist.pop(0)
        # Warm-up: not enough context yet → hold neutral.
        if len(self._flat_hist) < self.L_ctx:
            self._ego_inputs_hist.append(_NEUTRAL_ACTION.copy())
            if len(self._ego_inputs_hist) > self.L_ctx:
                self._ego_inputs_hist.pop(0)
            return action_vec_to_controller(_NEUTRAL_ACTION)
        # Transition: on the first inference call, _ego_inputs_hist is one
        # short of _flat_hist (warm-up appends inside its own branch). Pad
        # with neutral so both rolling buffers have L_ctx entries.
        if len(self._ego_inputs_hist) < self.L_ctx:
            self._ego_inputs_hist.append(_NEUTRAL_ACTION.copy())
        # Replan cadence: K frames if K>0, else L_chunk (open-loop).
        replan_period = self.K if self.K > 0 else self.L_chunk
        if self._pending is None or self._offset >= replan_period:
            raw = _live_batch_from_rolling(self._flat_hist, self._ego_inputs_hist, self.ego_prefix)
            batch = preprocess_inputs(raw, self.stats)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if self.K > 0:
                # Bootstrap: no prev chunk → zero bridge; bridge plays neutral
                # for the next K frames. Steady state: bridge = prev_chunk[:K].
                if self._pending is None:
                    new_bridge = np.zeros((self.K, A_DIM), dtype=np.float32)
                else:
                    new_bridge = self._pending[: self.K].astype(np.float32)
                bridge_t = torch.from_numpy(new_bridge).unsqueeze(0).to(self.device)
                self._pending = _integrate_chunk(self.model, batch, self.n_flow_steps, self.device, bridge=bridge_t)
                self._current_bridge = new_bridge
            else:
                self._pending = _integrate_chunk(self.model, batch, self.n_flow_steps, self.device)
            self._offset = 0
        # Played action: bridge if K>0 (next K to execute), else chunk directly.
        if self.K > 0:
            a = self._current_bridge[self._offset]
        else:
            a = self._pending[self._offset]
        self._offset += 1
        self._ego_inputs_hist.append(a.astype(np.float32))
        if len(self._ego_inputs_hist) > self.L_ctx:
            self._ego_inputs_hist.pop(0)
        return action_vec_to_controller(a)


@dataclass
class SelfPlayController:
    """Drive both ports of a self-play match from one model with a single
    batched forward pass per chunk boundary.

    Two `_SelfPlayPortView` instances (one per port) hold a reference back
    here and delegate per-frame. drive() calls each port's source in turn;
    the first call per frame advances state for BOTH ports (and, when a
    chunk boundary is hit, runs one batched forward of shape [2, L_ctx, ...]).
    The second call just reads the cached action.
    """

    model: nn.Module
    stats: dict
    L_ctx: int = L_CTX
    L_chunk: int = L_CHUNK
    K: int = 0
    n_flow_steps: int = 8
    device: str = DEVICE
    _ports: dict = field(
        default_factory=lambda: {
            "p1": {"flat_hist": [], "ego_inputs_hist": []},
            "p2": {"flat_hist": [], "ego_inputs_hist": []},
        }
    )
    _pending: dict = field(default_factory=lambda: {"p1": None, "p2": None})
    _current_bridge: dict = field(default_factory=lambda: {"p1": None, "p2": None})
    _offset: int = 0
    _last_frame_done: int = -1
    _last_actions: dict = field(default_factory=lambda: {"p1": _NEUTRAL_ACTION.copy(), "p2": _NEUTRAL_ACTION.copy()})

    def view(self, ego_prefix: Literal["p1", "p2"]) -> _SelfPlayPortView:
        return _SelfPlayPortView(coord=self, ego_prefix=ego_prefix)

    def _tick(self, ego_prefix: Literal["p1", "p2"], frame_index: int, last_gamestate: dict | None) -> np.ndarray:
        if frame_index != self._last_frame_done:
            self._advance(last_gamestate)
            self._last_frame_done = frame_index
        return self._last_actions[ego_prefix]

    def _advance(self, last_gamestate: dict | None) -> None:
        for ego in ("p1", "p2"):
            buf = self._ports[ego]
            if last_gamestate is not None:
                buf["flat_hist"].append(_flatten_canonical_frame(last_gamestate))
                if len(buf["flat_hist"]) > self.L_ctx:
                    buf["flat_hist"].pop(0)
        # Warm-up: hold neutral on both ports until both buffers are full.
        if any(len(self._ports[e]["flat_hist"]) < self.L_ctx for e in ("p1", "p2")):
            for ego in ("p1", "p2"):
                buf = self._ports[ego]
                buf["ego_inputs_hist"].append(_NEUTRAL_ACTION.copy())
                if len(buf["ego_inputs_hist"]) > self.L_ctx:
                    buf["ego_inputs_hist"].pop(0)
            self._last_actions = {ego: _NEUTRAL_ACTION.copy() for ego in ("p1", "p2")}
            return
        # Transition: pad ego_inputs_hist to L_ctx on the first inference frame.
        for ego in ("p1", "p2"):
            buf = self._ports[ego]
            if len(buf["ego_inputs_hist"]) < self.L_ctx:
                buf["ego_inputs_hist"].append(_NEUTRAL_ACTION.copy())
        # Replan cadence: K frames if K>0, else L_chunk (open-loop).
        replan_period = self.K if self.K > 0 else self.L_chunk
        if self._pending["p1"] is None or self._offset >= replan_period:
            stacked = self._build_stacked_batch()
            batch = preprocess_inputs(stacked, self.stats)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if self.K > 0:
                if self._pending["p1"] is None:
                    new_bridges = {ego: np.zeros((self.K, A_DIM), dtype=np.float32) for ego in ("p1", "p2")}
                else:
                    new_bridges = {ego: self._pending[ego][: self.K].astype(np.float32) for ego in ("p1", "p2")}
                bridge_stack = np.stack([new_bridges["p1"], new_bridges["p2"]], axis=0)
                bridge_t = torch.from_numpy(bridge_stack).to(self.device)
                plans = _integrate_chunk_batched(self.model, batch, self.n_flow_steps, self.device, bridge=bridge_t)
                self._pending = {"p1": plans[0], "p2": plans[1]}
                self._current_bridge = new_bridges
            else:
                plans = _integrate_chunk_batched(self.model, batch, self.n_flow_steps, self.device)
                self._pending = {"p1": plans[0], "p2": plans[1]}
            self._offset = 0
        actions: dict[str, np.ndarray] = {}
        for ego in ("p1", "p2"):
            if self.K > 0:
                a = self._current_bridge[ego][self._offset]
            else:
                a = self._pending[ego][self._offset]
            actions[ego] = a
            buf = self._ports[ego]
            buf["ego_inputs_hist"].append(a.astype(np.float32))
            if len(buf["ego_inputs_hist"]) > self.L_ctx:
                buf["ego_inputs_hist"].pop(0)
        self._offset += 1
        self._last_actions = actions

    def _build_stacked_batch(self) -> dict[str, np.ndarray]:
        """Stack p1- and p2-perspective rolling-buffer batches along batch dim."""
        per_ego: dict[str, dict[str, np.ndarray]] = {}
        for ego in ("p1", "p2"):
            buf = self._ports[ego]
            per_ego[ego] = _live_batch_from_rolling(buf["flat_hist"], buf["ego_inputs_hist"], ego_prefix=ego)
        return {k: np.concatenate([per_ego["p1"][k], per_ego["p2"][k]], axis=0) for k in per_ego["p1"]}


@dataclass(frozen=True)
class _SelfPlayPortView:
    coord: SelfPlayController
    ego_prefix: Literal["p1", "p2"]

    def __call__(self, frame_index: int, last_gamestate: dict | None):
        a = self.coord._tick(self.ego_prefix, frame_index, last_gamestate)
        return action_vec_to_controller(a)


def _last_finite_stock(arr: np.ndarray) -> int:
    """Last in-game stock value. Trailing IN_GAME → menu frames have NaN per-port
    fields, so int(arr[-1]) would either raise or silently report 0."""
    finite = arr[np.isfinite(arr)]
    return int(finite[-1]) if len(finite) > 0 else 0


def closed_loop_eval(model: nn.Module, max_frames: int = 3600) -> dict:
    """Drive one Fox-ditto vs cpu_level=9 on FD for up to `max_frames`.
    FFW + EXI input pipe via the exi-ai Dolphin build for speed; emulation_speed=0
    uncaps framerate. drive() returns the moment a match ends naturally, so
    max_frames is a safety cap, not a target."""
    import melee

    from hal.fixtures import DOLPHIN_EXIAI
    from hal.fixtures import ISO
    from hal.fixtures import ensure
    from hal.paths import EMULATOR_PATH
    from hal.sim.loop import drive
    from hal.sim.session import Matchup
    from hal.sim.session import PlayerSetup
    from hal.sim.session import Session
    from hal.sim.sources import InternalControllerSource

    ensure(DOLPHIN_EXIAI)
    matchup = Matchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX, cpu_level=0),
            PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=9),
        ),
    )
    sources = {
        1: ModelControllerSource(
            model=model,
            stats=stats,
            ego_prefix="p1",
            K=int(getattr(model, "K", 0)),
        ),
        2: InternalControllerSource(),
    }
    was_training = model.training
    model.eval()
    try:
        with Session(
            iso_path=ensure(ISO),
            dolphin_path=EMULATOR_PATH,
            blocking_input=True,
            use_exi_inputs=True,
            enable_ffw=True,
            emulation_speed=0.0,
        ) as s:
            traj = drive(s, matchup, sources, max_frames=max_frames)
    except Exception as e:
        print(f"[eval] closed-loop crashed: {e!r}", flush=True)
        if was_training:
            model.train()
        return dict(crashed=1.0)
    if was_training:
        model.train()
    p1_stock = _last_finite_stock(traj.post[1]["stock"])
    p2_stock = _last_finite_stock(traj.post[2]["stock"])
    p1_max_pct = float(np.nanmax(traj.post[1]["percent"]))
    p2_max_pct = float(np.nanmax(traj.post[2]["percent"]))
    return dict(
        stocks_taken=4 - p2_stock,
        stocks_lost=4 - p1_stock,
        damage_dealt=p2_max_pct,
        damage_taken=p1_max_pct,
        frames=len(traj),
    )


# %%
# Sanity 1 — preprocess a batched window from val.
K_LATENCY = int(cfg.get("latency_frames", 0))
# Skip the heavy sanity / training cells on plain `import notebooks.toy_train`
# (e.g. from pytest). They still run under `python notebooks/toy_train.py` and
# in VSCode "Run Cell" (which evaluates each cell with __name__ == '__main__').
if __name__ == "__main__":
    ds = StreamingDataset(local=str(Path(DATA_ROOT) / "val"), batch_size=1)
    val_loader = StreamingDataLoader(
        WindowSampler(ds, L_CTX, L_CHUNK, K=K_LATENCY),
        batch_size=2,
        num_workers=0,
        collate_fn=collate_windows,
    )
    raw = next(iter(val_loader))
    batch_t = preprocess_inputs(raw, stats)
    assert batch_t["ego_position_x"].shape == (2, L_CTX + K_LATENCY + L_CHUNK)
    assert batch_t["ego_action"].shape == (2, L_CTX + K_LATENCY + L_CHUNK)
    print("preprocess shapes OK")


# %%
# Sanity 2 — one forward + loss + backward.
if __name__ == "__main__":
    model = FlowMatchingPolicy(cfg).to(DEVICE)
    batch_d = _to_device(batch_t, DEVICE)
    _actions_all = stack_ego_actions(batch_d)
    bridge_d = _actions_all[:, L_CTX : L_CTX + K_LATENCY, :] if K_LATENCY > 0 else None
    a_target = _actions_all[:, L_CTX + K_LATENCY :, :]
    B = a_target.shape[0]
    t = torch.rand(B, device=DEVICE)
    z = torch.randn_like(a_target)
    t_b = t.view(B, 1, 1)
    a_t = (1 - t_b) * z + t_b * a_target
    v_target = a_target - z
    v_pred = model(batch_d, a_t, t, bridge=bridge_d)
    loss = F.mse_loss(v_pred, v_target)
    loss.backward()
    print(
        f"v_pred {tuple(v_pred.shape)} loss {loss.item():.4f}; n_params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )


# %%
# Sanity 3 — overfit a single window to ~0 loss (fixed t, z, dropout off).
# This validates the model has the capacity to fit; FM loss with fixed noise
# is deterministic given parameters, so it should go to ~0.
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    overfit_cfg = {**cfg, "dropout": 0.0}
    overfit_model = FlowMatchingPolicy(overfit_cfg).to(DEVICE)
    overfit_opt = AdamW(overfit_model.parameters(), lr=3e-4, betas=(0.9, 0.95))

    one_sample = ds[0]
    T = len(one_sample["frame"])
    _window_len = L_CTX + K_LATENCY + L_CHUNK
    assert _window_len <= T, f"replay too short for overfit ({T} < {_window_len})"
    window = {k: v[0:_window_len] for k, v in one_sample.items()}
    window = _relabel_ego(window, "p1")
    raw_single = {k: v[None, ...] for k, v in window.items()}
    single_batch = _to_device(preprocess_inputs(raw_single, stats), DEVICE)
    _single_actions = stack_ego_actions(single_batch)
    single_bridge = _single_actions[:, L_CTX : L_CTX + K_LATENCY, :] if K_LATENCY > 0 else None
    target_single = _single_actions[:, L_CTX + K_LATENCY :, :]

    fixed_t = torch.full((1,), 0.5, device=DEVICE)
    fixed_z = torch.randn(1, L_CHUNK, A_DIM, device=DEVICE)
    fixed_at = 0.5 * fixed_z + 0.5 * target_single
    fixed_v = target_single - fixed_z

    overfit_model.train()
    losses = []
    for step in range(800):
        v_pred = overfit_model(single_batch, fixed_at, fixed_t, bridge=single_bridge)
        loss = F.mse_loss(v_pred, fixed_v)
        overfit_opt.zero_grad()
        loss.backward()
        overfit_opt.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"overfit step {step}: loss {loss.item():.6f}")
    print(f"final overfit loss: {losses[-1]:.6f} (initial {losses[0]:.4f})")
    assert losses[-1] < 1e-3, f"single-episode overfit did not converge: {losses[-1]:.4f}"
    print("OK: single-episode overfit converged below 1e-3")


# %%
# Real training. Runs when this file is invoked as `python notebooks/toy_train.py`.
# Sanity cells above act as a quick preflight; the run then kicks off.
if __name__ == "__main__":
    train_model = FlowMatchingPolicy(cfg).to(DEVICE)
    train_loader = make_loader(cfg, "train")
    val_loader_long = make_loader({**cfg, "num_workers": 0}, "val")
    train(train_model, train_loader, cfg, DEVICE, val_loader_long, comment="lat4-15k")
