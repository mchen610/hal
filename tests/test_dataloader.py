"""WindowDataset determinism — the val set must be reproducible across runs,
and train windows must still vary across epochs given a fixed seed."""

import numpy as np

from hal.data.schema import SCHEMA_VERSION
from hal.training.dataloader import WindowDataset

L_CTX, L_CHUNK = 6, 4
_L = L_CTX + L_CHUNK


def _fake_mds(
    n_samples: int = 6,
    length: int = 60,
    character_pairs: list[tuple[int, int]] | None = None,
) -> list[dict[str, np.ndarray]]:
    """In-memory stand-in for a StreamingDataset: each sample is one replay."""
    out = []
    for i in range(n_samples):
        sample = {
            "schema_version": SCHEMA_VERSION,
            "frame": np.arange(length, dtype=np.int32),
            "p1_position_x": np.arange(length, dtype=np.float32),
            "p2_position_x": np.arange(length, dtype=np.float32) + 1000.0,
        }
        if character_pairs is not None:
            p1, p2 = character_pairs[i]
            sample["p1_character"] = np.full(length, p1, dtype=np.int32)
            sample["p2_character"] = np.full(length, p2, dtype=np.int32)
        out.append(sample)
    return out


def _fingerprint(sampler: WindowDataset) -> list[tuple[int, str]]:
    """(window start, ego side) per yielded window — observable proxy for the
    sampler's two random draws (start offset + ego_prefix)."""
    out = []
    for w in sampler:
        start = int(w["frame"][0])
        ego_side = "p1" if w["ego_position_x"][0] < 500 else "p2"
        out.append((start, ego_side))
    return out


def test_same_seed_same_windows() -> None:
    """Two fresh samplers with the same seed yield identical windows — this is
    what makes cached val loss comparable across runs."""
    a = _fingerprint(WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0))
    b = _fingerprint(WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0))
    assert a == b


def test_different_seed_different_windows() -> None:
    a = _fingerprint(WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0))
    b = _fingerprint(WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=1))
    assert a != b


def test_windows_vary_across_epochs() -> None:
    """A single sampler iterated twice (two epochs) draws different windows, so
    a fixed seed doesn't freeze train augmentation to one window per replay."""
    s = WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0)
    epoch0 = _fingerprint(s)
    epoch1 = _fingerprint(s)
    assert epoch0 != epoch1


def test_window_length_and_ctx_pad() -> None:
    """Every emitted window is exactly L_ctx + L_chunk frames and carries an int
    ctx_pad — the neutral [ctx | chunk] contract, no bridge frames."""
    for w in WindowDataset(_fake_mds(), L_CTX, L_CHUNK, seed=0):
        assert len(w["frame"]) == _L
        assert "ctx_pad" in w


def test_cold_start_floor_skips_too_short() -> None:
    """cs_min=1 needs >=1 real context frame and the L_chunk chunk in-episode, so
    a replay of exactly L_chunk frames (cs_max=0 < 1) yields nothing."""
    assert list(WindowDataset(_fake_mds(n_samples=2, length=L_CHUNK), L_CTX, L_CHUNK, seed=0)) == []
    # one extra frame is enough for a single anchor (cs=1, fully left-padded ctx).
    assert list(WindowDataset(_fake_mds(n_samples=2, length=L_CHUNK + 1), L_CTX, L_CHUNK, seed=0))


def test_character_pair_filter_keeps_matching_replays() -> None:
    rows = list(
        WindowDataset(
            _fake_mds(character_pairs=[(1, 1), (1, 22), (1, 1), (2, 1), (1, 1), (18, 18)]),
            L_CTX,
            L_CHUNK,
            seed=0,
            character_pair=(1, 1),
        )
    )
    assert len(rows) == 3
    assert all(set(row["ego_character"]) <= {0, 1} and set(row["opp_character"]) <= {0, 1} for row in rows)
    assert all(row["ego_character"][-1] == 1 and row["opp_character"][-1] == 1 for row in rows)
