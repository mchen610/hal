# HAL

Training superhuman AI for *Super Smash Bros. Melee* via imitation learning and RL.

Blog: https://ericyuegu.com/melee-pt1.

## Setup

Tested on Ubuntu 20.04+. Linux is highly recommended, macOS/Window support is dodgy.

### 1. Install tooling

```bash
# uv (Python + project manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Rust (for peppi-py — one-time compile via maturin)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"
```

**macOS only**: `libmelee` needs a system `enet`:

```bash
brew install enet
```

### 2. Clone and sync dependencies

```bash
git clone git@github.com:ericyuegu/hal.git
cd hal
uv sync                # ~35s for the peppi-py compile on first sync; cached after
```

On macOS pass the enet paths through:

```bash
brew install enet
CFLAGS="-I$(brew --prefix enet)/include" LDFLAGS="-L$(brew --prefix enet)/lib -lenet" uv sync
```

### 3. Set up Cloudflare R2 credentials

Integration fixtures (dev replay archive, pre-built MDS bundle, Melee ISO, Dolphin build) live in a private R2 bucket. Ask Eric for credentials.

```bash
cp .env.example .env
$EDITOR .env           # fill in AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BUCKET=hal
```

Then either use [direnv](https://direnv.net/) (`direnv allow`) or source manually each shell / add it as an alias in your ~/.bashrc:

```bash
set -a; source .env; set +a
```

### 4. Fetch fixtures

One command pulls everything you need into `<repo>/fixtures/`.

```bash
uv run python -m hal.scripts.fetch
```

After it finishes, `fixtures/` looks like:

```
fixtures/
  dev.7z                        # 37 MB — slp archive
  dev/mds/                      # train/, val/, test/, manifest.jsonl, stats.json
  ssbm.ciso
  dolphin/exiai/squashfs-root/  # AppRun + game files for the headless build
```

Re-running `fetch` is idempotent (logs `skip <name> (sha match)`). Fetch a single fixture with `--name {dev.7z | dev-mds | ssbm.ciso | dolphin-exiai}`. All paths are env-overridable (`HAL_ISO_PATH`, `HAL_EMULATOR_PATH`, `HAL_DEV_ARCHIVE`, `HAL_DEV_MDS_DIR`) if you keep fixtures elsewhere — see `hal/paths.py`.

### 5. Verify

```bash
uv run pytest -q tests/
```

Expected: **39 passed**. If you see `9 skipped` instead, fetch hasn't run yet (or the env vars aren't sourced).

A closed-loop bit-exact round-trip through Dolphin is the strongest end-to-end check:

```bash
uv run python -m hal.scripts.roundtrip --max-frames 200
# expect: PASS (bit-exact across 11 post-fields × 2 ports)
```

## Data pipeline

Three stages in `hal/scripts/` turn `.slp` files (loose or `.7z`-archived) into MDS shards:

1. **`index`** — walk replays; write `index.jsonl` (one row of metadata per file).
2. **`filter`** — query `index.jsonl`; emit `paths.txt` of replays to keep.
3. **`materialize`** — read `paths.txt` + `index.jsonl`; write MDS shards + `manifest.jsonl` + `stats.json`.

`paths.txt` lines are either filesystem paths or `archive://<abs-archive>!<member>` synthetic paths; one file may mix both, and the materialize stage streams archive members from a bounded tmpfs ring (`/dev/shm/...`) without ever extracting to disk.

```bash
# Stage 1 — index a .7z archive (no extraction; tmpfs-backed)
uv run python -m hal.scripts.index \
    --archive fixtures/dev.7z --output /tmp/index.jsonl

# Stage 2 — filter to a paths.txt (defaults: completed games, ≥1500 frames, six tournament stages)
uv run python -m hal.scripts.filter \
    --index /tmp/index.jsonl --output /tmp/paths.txt

# Stage 3 — paths.txt + index.jsonl → MDS shards
uv run python -m hal.scripts.materialize \
    --paths-file /tmp/paths.txt --index /tmp/index.jsonl --output /tmp/mds
```

Fold additional archives into one `index.jsonl` with `--incremental` on Stage 1.

## Closed-loop round-trip

The driver in `hal/sim/` plays an MDS row back through Dolphin and diffs against the original `.slp`:

```bash
uv run python -m hal.scripts.roundtrip --max-frames 200
```

Defaults read from `fixtures/`; override via flags. Training and evaluation drivers are being rewritten on top of `hal/sim/` and the new MDS schema; nothing here ships yet.
