"""Latency-aware (receding-horizon) chunked prediction for the toy
flow-matching policy.

K = ``cfg["latency_frames"]`` ≥ 0 frames. K>0 enables the bridge: the model
conditions on the K already-committed actions for frames ``[t, t+K)`` that
will execute while inference runs, and predicts the chunk for
``[t+K, t+K+L_chunk)``. K=0 reproduces the original open-loop architecture
exactly (backward compat for older checkpoints).

Tests cover (1) the train forward/backward shapes for both K, (2) the
inference rolling-buffer replan cadence, and (3) the bridge-feedback loop.
The full ``notebooks/toy_train.py`` is loaded via ``importlib`` so the
heavy ``if __name__ == "__main__"`` sanity cells stay silent.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
_NOTEBOOKS = _REPO / "notebooks"
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))


def _load_toy_train():
    """Import notebooks/toy_train.py as a fresh module.

    Module-level work loads dataset stats from disk (fast), but the heavy
    sanity cells are guarded so they don't fire on import.
    """
    spec = importlib.util.spec_from_file_location("toy_train_test", _NOTEBOOKS / "toy_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


toy_train = _load_toy_train()
FlowMatchingPolicy = toy_train.FlowMatchingPolicy
ModelControllerSource = toy_train.ModelControllerSource
A_DIM = toy_train.A_DIM
ACTION_CHANNELS = toy_train.ACTION_CHANNELS
CAT_FEATURES = toy_train.CAT_FEATURES
FLOAT_FEATURES = toy_train.FLOAT_FEATURES


def _tiny_cfg(L_ctx: int = 4, L_chunk: int = 8, K: int = 4) -> dict:
    """Smallest viable FlowMatchingPolicy config for fast unit tests."""
    return dict(
        d_model=16,
        n_layers=1,
        n_heads=2,
        dim_feedforward=32,
        dropout=0.0,
        time_emb_dim=8,
        L_ctx=L_ctx,
        L_chunk=L_chunk,
        latency_frames=K,
        ego_history_dropout_prob=0.0,
    )


def _fake_batch(L_ctx: int, K: int, L_chunk: int, B: int = 2) -> dict[str, torch.Tensor]:
    """Build a preprocessed-style batch dict matching the schema the model expects.
    Includes both ego and opp side, all FLOAT_FEATURES + masks + categoricals +
    action_frame, plus per-channel ego_<action_channel> columns spanning
    ``L_ctx + K + L_chunk`` frames.
    """
    L = L_ctx + K + L_chunk
    batch: dict[str, torch.Tensor] = {}
    for side in ("ego", "opp"):
        for f in FLOAT_FEATURES:
            batch[f"{side}_{f}"] = torch.randn(B, L)
        batch[f"{side}_action_frame"] = torch.rand(B, L)
        for cat, (vocab, _) in CAT_FEATURES.items():
            batch[f"{side}_{cat}"] = torch.randint(0, vocab, (B, L), dtype=torch.long)
    for ch in ACTION_CHANNELS:
        # Channel values in [-1, 1] (buttons get rounded to 0/1 inside model anyway).
        batch[f"ego_{ch}"] = torch.rand(B, L) * 2.0 - 1.0
    return batch


# --------------------------------------------------------------------------- #
# Test 1 — forward + backward shape, K=4 and K=0 backward compat.
# --------------------------------------------------------------------------- #
def test_train_forward_backward_both_k() -> None:
    torch.manual_seed(0)
    for K in (4, 0):
        cfg = _tiny_cfg(K=K)
        L_ctx, L_chunk = cfg["L_ctx"], cfg["L_chunk"]
        model = FlowMatchingPolicy(cfg)
        batch = _fake_batch(L_ctx, K, L_chunk, B=2)
        actions_all = torch.stack([batch[f"ego_{c}"] for c in ACTION_CHANNELS], dim=-1)
        bridge = actions_all[:, L_ctx : L_ctx + K, :] if K > 0 else None
        target = actions_all[:, L_ctx + K :, :]
        B = target.shape[0]
        t = torch.rand(B)
        z = torch.randn_like(target)
        a_t = (1 - t.view(B, 1, 1)) * z + t.view(B, 1, 1) * target
        v_target = target - z
        v_pred = model(batch, a_t, t, bridge=bridge)
        assert v_pred.shape == (B, L_chunk, A_DIM), f"K={K}: got {tuple(v_pred.shape)}"
        loss = F.mse_loss(v_pred, v_target)
        loss.backward()
        assert torch.isfinite(loss).item(), f"K={K}: loss not finite"
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all().item(), f"K={K}: NaN grad in {name}"


# --------------------------------------------------------------------------- #
# Test 2 — inference rolling-buffer replan cadence.
# --------------------------------------------------------------------------- #
def _make_source_with_stubbed_forward(K: int, L_ctx: int, L_chunk: int):
    """ModelControllerSource whose forward is replaced by a counter-based stub.
    Each replan emits a unique chunk (chunk #i has every element = i + 0.001 * j
    for frame offset j) so tests can read back which chunk produced which action."""
    cfg = _tiny_cfg(L_ctx=L_ctx, L_chunk=L_chunk, K=K)
    model = FlowMatchingPolicy(cfg)
    state = {"call_count": 0, "bridges_seen": []}

    def fake_integrate_chunk(_model, _batch, _n_steps, _device, bridge=None):
        i = state["call_count"]
        state["call_count"] += 1
        if bridge is not None:
            state["bridges_seen"].append(bridge.detach().cpu().numpy()[0].copy())
        else:
            state["bridges_seen"].append(None)
        # Values stay in [-1, 1] so action_vec_to_controller's clip is a no-op.
        # chunk #i row j: 0.05*i + 0.001*j → unique per (i, j).
        chunk = np.zeros((L_chunk, A_DIM), dtype=np.float32)
        for j in range(L_chunk):
            chunk[j, :] = 0.05 * (i + 1) + 0.001 * j
        return chunk

    src = ModelControllerSource(
        model=model,
        stats={},  # preprocess_inputs is patched out below
        ego_prefix="p1",
        L_ctx=L_ctx,
        L_chunk=L_chunk,
        K=K,
        device="cpu",
    )
    return src, model, state, fake_integrate_chunk


def _dummy_gamestate() -> dict:
    """Minimal gamestate stub satisfying _flatten_canonical_frame."""
    post = dict(
        position={"x": 0.0, "y": 0.0},
        percent=0.0,
        shield=60.0,
        stock=4,
        direction=1.0,
        action=14,
        state_age=0.0,
        hitlag_left=0.0,
        jumps_used=0,
        airborne=0,
        hurtbox_state=0,
    )
    leader = {"post": post}
    ports = {1: {"leader": leader}, 2: {"leader": leader}}
    return {"ports": ports}


def test_inference_rolling_buffer_replan() -> None:
    torch.manual_seed(0)
    L_ctx, L_chunk, K = 4, 8, 4
    src, _model, state, fake_integrate = _make_source_with_stubbed_forward(K, L_ctx, L_chunk)
    # Warm-up ends at the first frame where flat_hist reaches L_ctx entries,
    # which is frame t = L_ctx - 1 (0-indexed). Replans then fire every K frames.
    first_replan_frame = L_ctx - 1
    # 3 replans → frames [first_replan_frame, first_replan_frame + K, + 2K), and
    # at least one trailing played-from-bridge frame to exercise the steady state.
    n_frames = first_replan_frame + 2 * K + 1  # 3 replans, +1 played frame
    played: list[np.ndarray] = []
    with (
        patch.object(toy_train, "_integrate_chunk", side_effect=fake_integrate),
        patch.object(
            toy_train,
            "preprocess_inputs",
            side_effect=lambda raw, _stats: {k: torch.from_numpy(v) for k, v in raw.items()},
        ),
    ):
        for t in range(n_frames):
            ctrl = src(frame_index=t, last_gamestate=_dummy_gamestate())
            played.append(np.array([ctrl.main_x, ctrl.main_y, ctrl.c_x, ctrl.c_y, ctrl.trigger_l, ctrl.trigger_r]))
    # Warm-up: frames before first_replan_frame produce neutral.
    for t in range(first_replan_frame):
        assert np.allclose(played[t], 0.0), f"frame {t}: expected neutral, got {played[t]}"
    # 3 replans expected.
    assert state["call_count"] == 3, f"expected 3 replans, got {state['call_count']}"
    # 1st replan: bootstrap → zero bridge.
    assert np.allclose(state["bridges_seen"][0], 0.0), "bootstrap bridge must be zeros"
    # Bridge at replan #n (n>=1) = chunk #(n-1)'s first K predictions.
    for n in (1, 2):
        prev_chunk_first_K = np.array(
            [[0.05 * n + 0.001 * j] * A_DIM for j in range(K)],
            dtype=np.float32,
        )
        assert np.allclose(state["bridges_seen"][n], prev_chunk_first_K, atol=1e-5), (
            f"replan {n}: bridge mismatch\n got: {state['bridges_seen'][n]}\n want: {prev_chunk_first_K}"
        )
    # Played values for the K bootstrap frames at [first_replan_frame, +K) — neutral.
    for j in range(K):
        assert np.allclose(played[first_replan_frame + j], 0.0), (
            f"bootstrap played frame {first_replan_frame + j} must be neutral"
        )
    # Played values for the K frames after replan #2 — chunk #1's first K.
    for j in range(K):
        expected = 0.05 * 1 + 0.001 * j
        assert played[first_replan_frame + K + j][0] == pytest.approx(expected, abs=1e-5)
    # Played values right after replan #3 — chunk #2's first K (one trailing frame asserted).
    assert played[first_replan_frame + 2 * K][0] == pytest.approx(0.05 * 2 + 0.001 * 0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Test 3 — bridge correctness: bridge[N] equals chunk[N-1][:K].
# --------------------------------------------------------------------------- #
def test_bridge_feedback_matches_prev_chunk() -> None:
    torch.manual_seed(0)
    L_ctx, L_chunk, K = 4, 8, 4
    src, _model, state, fake_integrate = _make_source_with_stubbed_forward(K, L_ctx, L_chunk)
    chunks: list[np.ndarray] = []

    def fake_with_capture(_model, _batch, _n_steps, _device, bridge=None):
        c = fake_integrate(_model, _batch, _n_steps, _device, bridge=bridge)
        chunks.append(c.copy())
        return c

    first_replan_frame = L_ctx - 1
    n_replans = 4
    n_frames = first_replan_frame + (n_replans - 1) * K + 1
    with (
        patch.object(toy_train, "_integrate_chunk", side_effect=fake_with_capture),
        patch.object(
            toy_train,
            "preprocess_inputs",
            side_effect=lambda raw, _stats: {k: torch.from_numpy(v) for k, v in raw.items()},
        ),
    ):
        for t in range(n_frames):
            src(frame_index=t, last_gamestate=_dummy_gamestate())
    assert len(chunks) == n_replans, f"expected {n_replans} replans, got {len(chunks)}"
    # Bridge invariant: for N >= 1, bridges_seen[N] == chunks[N-1][:K].
    for N in range(1, n_replans):
        assert np.allclose(state["bridges_seen"][N], chunks[N - 1][:K]), (
            f"replan {N}: bridge != prev chunk first K\n bridge={state['bridges_seen'][N]}\n prev[:K]={chunks[N - 1][:K]}"
        )
