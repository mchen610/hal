from pathlib import Path

import pytest

from hal.eval.harness import local_session_cfg


def test_local_session_cfg_uses_explicit_paths(tmp_path: Path) -> None:
    iso = tmp_path / "ssbm.ciso"
    dolphin = tmp_path / "Dolphin"
    replay_dir = tmp_path / "replays"
    iso.touch()
    dolphin.touch()

    cfg = local_session_cfg(replay_dir, iso_path=iso, dolphin_path=dolphin)

    assert cfg.iso_path == iso
    assert cfg.dolphin_path == dolphin
    assert cfg.replay_dir == str(replay_dir)
    assert not cfg.use_exi_inputs
    assert not cfg.enable_ffw
    assert cfg.emulation_speed == 1.0
    assert not cfg.blocking_input
    assert cfg.tmp_home_directory


def test_local_session_cfg_rejects_missing_explicit_dolphin(tmp_path: Path) -> None:
    iso = tmp_path / "ssbm.ciso"
    iso.touch()

    with pytest.raises(FileNotFoundError):
        local_session_cfg(iso_path=iso, dolphin_path=tmp_path / "missing-dolphin")


def test_local_session_cfg_rejects_missing_explicit_iso(tmp_path: Path) -> None:
    dolphin = tmp_path / "Dolphin"
    dolphin.touch()

    with pytest.raises(FileNotFoundError):
        local_session_cfg(iso_path=tmp_path / "missing.ciso", dolphin_path=dolphin)
