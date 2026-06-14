"""CLI: summarize character matchup distribution from a replay manifest.

Defaults to the R2-hosted processed dataset in ``hal.streams``:

    uv run matchups --top 20
    uv run matchups --output /tmp/matchups.tsv

For local manifests:

    uv run matchups --manifest data/processed/dev/mds/manifest.jsonl
"""

import csv
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import TextIO

import melee
import tyro

from hal import r2
from hal.streams import BY_NAME
from hal.streams import RANKED_ANONYMIZED_1

_DEFAULT_SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class MatchupRow:
    rank: int
    character_a: int
    character_b: int
    count: int
    percent: float
    by_split: dict[str, int]


@dataclass(frozen=True, slots=True)
class Summary:
    total_rows: int
    valid_rows: int
    skipped_rows: int
    rows: list[MatchupRow]
    character_counts: Counter[int]
    split_counts: Counter[str]


@dataclass(frozen=True, slots=True)
class Config:
    """Summarize unordered character matchups from a HAL manifest."""

    manifest: Path | None = None
    """Local manifest.jsonl. If omitted, stream from --dataset in R2."""

    dataset: str = RANKED_ANONYMIZED_1.name
    """Named R2 stream source from hal.streams."""

    output: Path | None = None
    """Optional TSV path for the complete matchup table."""

    top: int = 40
    """Rows to print to stdout. Use 0 to print only the summary."""

    ordered: bool = False
    """Keep player order. Default folds A-vs-B and B-vs-A together."""

    env_file: Path | None = Path(".env")
    """Optional dotenv-style file to load before R2 access."""


def _load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def _split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {uri!r}")
    bucket, _, key = uri[len("s3://") :].partition("/")
    if not bucket or not key:
        raise ValueError(f"expected s3://bucket/key URI, got {uri!r}")
    return bucket, key


def _remote_manifest_lines(dataset: str) -> Iterator[bytes]:
    if dataset not in BY_NAME:
        raise SystemExit(f"unknown dataset {dataset!r}; known: {sorted(BY_NAME)}")
    src = BY_NAME[dataset]
    bucket, key = _split_s3_uri(src.remote)
    obj = r2.client().get_object(Bucket=bucket, Key=f"{key}/manifest.jsonl")
    yield from obj["Body"].iter_lines()


def _local_manifest_lines(path: Path) -> Iterator[str]:
    with path.open() as f:
        yield from f


def _manifest_entries(lines: Iterable[str | bytes]) -> Iterator[dict[str, Any]]:
    for lineno, raw in enumerate(lines, start=1):
        line = raw.decode() if isinstance(raw, bytes) else raw
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid manifest JSON on line {lineno}: {e}") from e
        if not isinstance(entry, dict):
            raise ValueError(f"manifest line {lineno} is not a JSON object")
        yield entry


def _entry_split(entry: dict[str, Any]) -> str:
    annotation = entry.get("annotation")
    if isinstance(annotation, dict):
        split = annotation.get("split")
        if isinstance(split, str) and split:
            return split
    return "unassigned"


def _entry_matchup(entry: dict[str, Any], *, ordered: bool) -> tuple[int, int] | None:
    players = entry.get("players")
    if not isinstance(players, list) or len(players) != 2:
        return None
    chars: list[int] = []
    for player in players:
        if not isinstance(player, dict) or "character" not in player:
            return None
        chars.append(int(player["character"]))
    return (chars[0], chars[1]) if ordered else tuple(sorted(chars))


def summarize(entries: Iterable[dict[str, Any]], *, ordered: bool = False) -> Summary:
    matchups: Counter[tuple[int, int]] = Counter()
    by_split: dict[tuple[int, int], Counter[str]] = {}
    character_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    total = 0
    skipped = 0

    for entry in entries:
        total += 1
        matchup = _entry_matchup(entry, ordered=ordered)
        if matchup is None:
            skipped += 1
            continue
        split = _entry_split(entry)
        matchups[matchup] += 1
        by_split.setdefault(matchup, Counter())[split] += 1
        split_counts[split] += 1
        character_counts.update(matchup)

    valid = sum(matchups.values())
    rows = [
        MatchupRow(
            rank=rank,
            character_a=matchup[0],
            character_b=matchup[1],
            count=count,
            percent=(count / valid * 100.0) if valid else 0.0,
            by_split=dict(by_split[matchup]),
        )
        for rank, (matchup, count) in enumerate(matchups.most_common(), start=1)
    ]
    return Summary(
        total_rows=total,
        valid_rows=valid,
        skipped_rows=skipped,
        rows=rows,
        character_counts=character_counts,
        split_counts=split_counts,
    )


def _character_name(character: int) -> str:
    try:
        return melee.Character(character).name
    except ValueError:
        return f"UNKNOWN_{character}"


def _split_names(summary: Summary) -> list[str]:
    seen = set(summary.split_counts)
    names = [split for split in _DEFAULT_SPLITS if split in seen]
    names.extend(sorted(seen - set(names)))
    return names


def _write_tsv(path: Path, summary: Summary) -> None:
    splits = _split_names(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "rank",
                "character_a_id",
                "character_a",
                "character_b_id",
                "character_b",
                "count",
                "percent",
                *splits,
            ]
        )
        for row in summary.rows:
            writer.writerow(
                [
                    row.rank,
                    row.character_a,
                    _character_name(row.character_a),
                    row.character_b,
                    _character_name(row.character_b),
                    row.count,
                    f"{row.percent:.4f}",
                    *(row.by_split.get(split, 0) for split in splits),
                ]
            )


def _print_summary(summary: Summary, *, top: int, fout: TextIO = sys.stdout) -> None:
    splits = _split_names(summary)
    print(f"rows: {summary.total_rows:,}", file=fout)
    print(f"valid_1v1_rows: {summary.valid_rows:,}", file=fout)
    print(f"skipped_rows: {summary.skipped_rows:,}", file=fout)
    print(f"unique_matchups: {len(summary.rows):,}", file=fout)
    print(f"unique_characters: {len(summary.character_counts):,}", file=fout)
    print("splits: " + ", ".join(f"{split}={summary.split_counts[split]:,}" for split in splits), file=fout)
    if top <= 0:
        return

    print("", file=fout)
    header = ["rank", "matchup", "count", "percent", *splits]
    print("\t".join(header), file=fout)
    for row in summary.rows[:top]:
        matchup = f"{_character_name(row.character_a)} vs {_character_name(row.character_b)}"
        fields = [
            str(row.rank),
            matchup,
            str(row.count),
            f"{row.percent:.2f}%",
            *(str(row.by_split.get(split, 0)) for split in splits),
        ]
        print("\t".join(fields), file=fout)


def run(cfg: Config) -> None:
    _load_env_file(cfg.env_file)
    lines = _local_manifest_lines(cfg.manifest) if cfg.manifest is not None else _remote_manifest_lines(cfg.dataset)
    summary = summarize(_manifest_entries(lines), ordered=cfg.ordered)
    _print_summary(summary, top=cfg.top)
    if cfg.output is not None:
        _write_tsv(cfg.output, summary)
        print(f"\nwrote {cfg.output}")


def main() -> None:
    run(tyro.cli(Config))


if __name__ == "__main__":
    main()
