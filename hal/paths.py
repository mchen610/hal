import os
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _REPO_ROOT / "fixtures"


def _env_path(var: str, default: Path) -> str:
    return os.getenv(var, str(default))


REPO_DIR: Final[str] = str(_REPO_ROOT)
ISO_PATH: Final[str] = _env_path("HAL_ISO_PATH", _FIXTURES / "ssbm.ciso")
EMULATOR_PATH: Final[str] = _env_path(
    "HAL_EMULATOR_PATH", _FIXTURES / "dolphin" / "exiai" / "squashfs-root" / "AppRun"
)
DEV_ARCHIVE_PATH: Final[str] = _env_path("HAL_DEV_ARCHIVE", _FIXTURES / "dev.7z")
DEV_MDS_DIR: Final[str] = _env_path("HAL_DEV_MDS_DIR", _FIXTURES / "dev" / "mds")
