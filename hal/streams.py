"""Cloud-streamed training datasets.

Where `hal/fixtures.py` mirrors small dev artifacts to disk and verifies
sha256, `streams.py` names training-scale MDS datasets that are too big
to fully materialize. The MosaicML `streaming` library handles
download-on-demand: shards are pulled into `local` as the dataloader
reads them, and the cache can be evicted under pressure.

Usage:

    from streaming import StreamingDataset
    from hal.streams import RANKED_ANONYMIZED_1

    remote, local = RANKED_ANONYMIZED_1.for_split("train")
    ds = StreamingDataset(remote=remote, local=str(local), batch_size=...)

Credentials come from the same env vars as `hal/fixtures.py`:
`AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. boto3 —
and therefore streaming — pick them up automatically; `s3://hal/...` URIs
resolve against R2's endpoint with no further configuration.

Cache layout mirrors the R2 prefix: `<repo>/data/<remote-key-path>/<split>/`,
already gitignored via `/data/`. Treat the cache as streaming-managed.
To pre-warm before going offline, iterate the dataset once end-to-end.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from loguru import logger

from hal import r2
from hal.paths import REPO_DIR


@dataclass(frozen=True, slots=True)
class StreamSource:
    """One MDS dataset with `{train, val, test}/` splits served from R2.

    `remote` is the s3:// URI of the MDS root; `local` is its cache mirror
    relative to repo root. `for_split(name)` returns the (remote, local)
    pair ready to drop into `StreamingDataset`.
    """

    name: str
    remote: str
    local: Path

    def for_split(self, split: str) -> tuple[str, Path]:
        return f"{self.remote}/{split}", Path(REPO_DIR) / self.local / split

    @property
    def local_root(self) -> Path:
        return Path(REPO_DIR) / self.local


RANKED_ANONYMIZED_1: Final[StreamSource] = StreamSource(
    name="ranked-anonymized-1",
    remote="s3://hal/processed/ranked-anonymized-1/mds",
    local=Path("data/processed/ranked-anonymized-1/mds"),
)

ALL: Final[tuple[StreamSource, ...]] = (RANKED_ANONYMIZED_1,)
BY_NAME: Final[dict[str, StreamSource]] = {s.name: s for s in ALL}
# Reverse map: local cache root (string) -> remote URI. Lets the dataloader turn
# a plain `data_root` into its R2 origin, while purely-local paths (dev MDS,
# overfit scratch) that aren't registered here resolve to None and stay local.
_REMOTE_BY_LOCAL: Final[dict[str, str]] = {str(s.local_root): s.remote for s in ALL}


def remote_for_local(local: str | Path) -> str | None:
    """R2 remote URI backing a local cache root, or None if it's local-only."""
    return _REMOTE_BY_LOCAL.get(str(Path(local) if Path(local).is_absolute() else Path(REPO_DIR) / local))


def _split_uri(remote: str) -> tuple[str, str]:
    """`s3://bucket/key/path` -> ('bucket', 'key/path')."""
    if not remote.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI, got {remote!r}")
    bucket, _, key = remote[len("s3://") :].partition("/")
    return bucket, key


def pull_dataset(src: StreamSource, *, max_workers: int = 16) -> int:
    """Eagerly mirror every object under the dataset's R2 prefix into the local cache.

    StreamingDataset pulls per-split shards on demand, which lets a fast GPU throttle
    on lazy per-shard R2 downloads during the first pass; the root ``stats.json`` (which
    sits outside any split, so the streaming layer never fetches it) is also needed
    before the first batch. This dataset is small (~20 GB zstd), so the cloud setup
    downloads the whole tree — shards + ``stats.json`` — up front. With every shard
    already on disk the loader never blocks on the network; StreamingDataset still owns
    decompression and eviction. Parallel + idempotent (an object whose local copy already
    matches the remote size is skipped), so a ``--resume`` reuses what the box pulled.
    Returns the number of objects actually downloaded.
    """
    bucket, key = _split_uri(src.remote)
    client = r2.client()
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{key}/")
    objs = [o for page in pages for o in page.get("Contents", ()) if not o["Key"].endswith("/")]

    def pull(obj: dict) -> bool:
        dest = src.local_root / obj["Key"][len(key) + 1 :]
        if dest.is_file() and dest.stat().st_size == obj["Size"]:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, obj["Key"], str(dest))
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        downloaded = sum(pool.map(pull, objs))
    logger.info(f"[streams] {src.name}: mirrored {len(objs)} objects ({downloaded} pulled) -> {src.local_root}")
    return downloaded
