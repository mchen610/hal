import os
from pathlib import Path
from typing import Final

# Don't move this file or this might break since it assumes grandparent is repo root
REPO_DIR: Final[str] = str(Path(__file__).resolve().parent.parent)

DOLPHIN_HOME_PATH: Final[str] = "/opt/slippi/Dolphin"
EMULATOR_PATH: Final[str] = "/opt/projects/hal2/emulator/squashfs-root/AppRun.wrapped"
ISO_PATH: Final[str] = "/opt/slippi/ssbm.ciso"

EVAL_REPLAY_DIR: Final[str] = "/opt/projects/hal2/replays"
# torch._dynamo.config.suppress_errors = True

MAC_EMULATOR_PATH: Final[str] = f"{Path.home()}/Library/Application Support/Slippi Launcher/netplay"
MAC_CISO_PATH: Final[str] = os.environ["HAL_MAC_CISO_PATH"]
MAC_REPLAY_DIR: Final[str] = f"{REPO_DIR}/replays"
