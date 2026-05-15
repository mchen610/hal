# HAL

Training superhuman AI for *Super Smash Bros. Melee*.

This project is under active development and is not ready for public use.

Blog post: https://ericyuegu.com/melee-pt1

## Setup

Python ≥ 3.14 on Ubuntu 20.04+. Dependencies are managed by [uv](https://docs.astral.sh/uv/). `peppi-py` is built from source via `maturin`, so a Rust toolchain is required.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"
uv sync                                            # peppi-py compile ~35s; cached after
```

macOS additionally needs `enet`:

```bash
brew install enet
CFLAGS="-I$(brew --prefix enet)/include" LDFLAGS="-L$(brew --prefix enet)/lib -lenet" uv sync
```

## Fetching integration fixtures

The dev replay archive, pre-built MDS bundle, ISO, and Dolphin (exi-ai) build are pulled from a private Cloudflare R2 bucket plus the upstream GitHub release. Credentials are out-of-band — ask Eric.

```bash
cp .env.example .env && $EDITOR .env               # fill in R2_* creds
uv run python -m hal.scripts.fetch                 # one-time; ~2 GB into <repo>/fixtures/
uv run pytest -q tests/                            # 39 tests; integration tests now run
```

Re-running `fetch` no-ops when local sha256 matches. Fetch one at a time with `--name dev.7z | dev-mds | ssbm.ciso | dolphin-exiai`. All paths are env-overridable (`HAL_ISO_PATH`, `HAL_EMULATOR_PATH`, `HAL_DEV_ARCHIVE`, `HAL_DEV_MDS_DIR`) if you keep fixtures elsewhere — see `hal/paths.py`.

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

You can fold additional archives into one `index.jsonl` with `--incremental` on Stage 1.

## Closed-loop round-trip

The closed-loop driver in `hal/sim/` plays an MDS row back through Dolphin and diffs against the original `.slp`:

```bash
uv run python -m hal.scripts.roundtrip --max-frames 200
# expect: PASS (bit-exact across 11 post-fields × 2 ports)
```

Defaults read from `fixtures/`; override via flags. Training and evaluation drivers are being rewritten on top of `hal/sim/` and the new MDS schema; nothing here ships yet.
