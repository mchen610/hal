# TODO: productionize hal/

Status: Part 1 (bug fixes + test coverage + interface polish) is being landed
inline. Part 2 — making the repo cloneable, runnable, and trainable in cloud
infra by a new collaborator in a few commands — is captured here.

The aim: a fresh checkout reaches green `pytest -m "not integration"` in
under 15 minutes with `make bootstrap && make test`.

## 1. Dolphin AppImage via GitHub release

Replace the current "manual download from somewhere + extract" story (per
README.md + scripts/setup.sh) with a pinned GitHub-release download.

- Upstream: `vladfi/slippi-Ishiiruka` (or whichever fork ships the headless
  ExiAI build). Confirm release tag + asset filename.
- Drop the LFS angle (current memory `project_dolphin_setup.md` says LFS;
  switch to release-pinned curl — no quota, version explicit in script,
  no `git lfs install` step for clone).
- `scripts/bootstrap.sh` rewrite:
  - Pin tag via `HAL_SLIPPI_RELEASE_TAG` env (default to known-good).
  - `curl -L` the AppImage asset; verify sha256 against
    `scripts/bootstrap_manifest.toml`.
  - Run `--appimage-extract` into `$HAL_DATA_HOME/dolphin/exiai/squashfs-root`.
  - `uv sync`.
  - Drop the `aws s3 cp "$SSBM_ISO_PATH"` step entirely; ISO acquisition
    is user-responsibility (legal). Just check presence and print a
    pointer if missing.
- `scripts/bootstrap_manifest.toml` (new): pins
  `slippi_release_tag`, `appimage_sha256`, optional `gale01r2_ini_sha256`.

## 2. Datasets via Cloudflare R2

### Bucket layout
```
r2://hal-datasets/<stream_name>/
  manifest.jsonl          # frozen, content-hashed (SCHEMA_VERSION already lands in Part 1)
  schema_v<N>.json        # column dtypes for shards below
  train/index.json + *.mds
  val/  ...
  test/ ...

r2://hal-datasets-raw/<stream_name>.7z   # raw .slp bundle, only for re-extraction
```

### `hal/data/streams.py` (Part 1 stubs this; Part 2 fleshes it out)
- `StreamRegistry` with named entries:
  - `melee_public_v2` (public ranked-anonymized + public dataset v2)
  - `dev` (small fixture for round-trip tests)
- Each entry resolves to a list of `streaming.Stream(local=..., remote=...)`
  instances for the training dataloader, plus a manifest URL and the
  `SCHEMA_VERSION` it was built with.
- `hal-fetch-manifest <stream>` CLI: downloads manifest.jsonl from R2 to
  `$HAL_DATA_HOME/processed/<stream>/manifest.jsonl` for local introspection /
  filter chaining.

### What "frozen index" buys us (for the README)
- **Training**: `(split, mds_row_idx)` maps MDS rows back to the source
  .slp's metadata (characters, rank, slp version). `DataConfig` filters
  at training time use this.
- **Round-trip**: `hal.emulator.roundtrip` picks an entry, reads the
  matching MDS row, replays it through Dolphin.
- **Reproducibility**: publishing "trained on `melee_public_v2`
  manifest sha256 9a3f…" lets another contributor verify identical splits.
  With `SCHEMA_VERSION` (Part 1, landed) the column set is also pinned.
- **Filter chains**: a downstream filter (e.g. only Fox-dittos) produces
  a subset manifest; the same row IDs work against the same MDS shards.

### Credentials
- `HAL_R2_ENDPOINT`, `HAL_R2_ACCESS_KEY_ID`, `HAL_R2_SECRET_ACCESS_KEY`.
- Public read-only access for `hal-datasets/*` (presigned URLs would also
  work for non-collaborators).

## 3. Contributor setup story

- `Makefile` (new):
  - `bootstrap` → `scripts/bootstrap.sh` (system deps + uv sync + Dolphin)
  - `bootstrap-dev` → fetch `dev` stream from R2 to `$HAL_DATA_HOME/processed`
  - `test` → `uv run pytest -m "not integration"`
  - `test-integration` → `uv run pytest -m integration` (requires ISO +
    Dolphin)
  - `lint` → `uv run ruff format --check . && uv run ruff check . && uv
    run ty check .`
  - `format` → `uv run ruff format . && uv run ruff check . --fix`
  - `check` → `lint && test`
- `CONTRIBUTING.md` (new): five-step path — clone → `make bootstrap` →
  optionally `make bootstrap-dev` → `make test` → run a notebook. Call
  out the legal ISO carve-out explicitly.
- `.env.example` additions: `HAL_SLIPPI_RELEASE_TAG`, `HAL_R2_*`,
  `HAL_DEV_ARCHIVE` (Part 1, landed), `HAL_DEV_MDS_DIR` (Part 1,
  landed), `HAL_RUN_REPLAY_REPRODUCTION_INTEGRATION`, `WANDB_API_KEY`.
- README.md trim: replace inline install with "run `make bootstrap`";
  link CONTRIBUTING.

## 4. CI (lint + unit only)

- `.github/workflows/ci.yml`: ubuntu-latest, Python 3.14, on push + PR:
  - `uv sync --dev`
  - `uv run ruff format --check .`
  - `uv run ruff check .`
  - `uv run ty check .`
  - `uv run pytest tests/ -m "not integration"`
- No ISO or Dolphin fixtures in CI. Integration tests stay opt-in /
  local (or a self-hosted runner later).

## 5. Stale / leftover cleanup

- `notebooks/` import errors: `hal/preprocess/transformations.py:23`
  imports `hal.data.stats` which no longer exists. The whole
  `hal/preprocess/` tree is legacy (per CLAUDE.md "delete liberally").
  Decide: revive `stats.py` for the new pipeline, or delete the legacy
  preprocess + notebooks that depend on it.
- `scripts/setup.sh` becomes obsolete once `scripts/bootstrap.sh`
  lands — delete it in the same PR to avoid drift.

## 6. Dependency story (defer, but track)

- All runtime deps unpinned in pyproject.toml; reproducibility relies
  on `uv.lock`. That's fine while iterating, but lock critical deps
  (torch, mosaicml-streaming, melee branch, peppi-py branch) once the
  data + model format stabilizes.
- Rust toolchain currently required for peppi-py git build; check if a
  pre-built wheel is available on the upstream release page and skip
  the rustc dep if so.
