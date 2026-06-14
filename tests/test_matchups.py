from pathlib import Path

from hal.scripts.matchups import _write_tsv
from hal.scripts.matchups import summarize


def test_summarize_folds_unordered_matchups() -> None:
    entries = [
        {
            "players": [{"character": 1}, {"character": 18}],
            "annotation": {"split": "train"},
        },
        {
            "players": [{"character": 18}, {"character": 1}],
            "annotation": {"split": "val"},
        },
        {
            "players": [{"character": 1}, {"character": 1}],
            "annotation": {"split": "train"},
        },
        {"players": [{"character": 1}]},
    ]

    summary = summarize(entries)

    assert summary.total_rows == 4
    assert summary.valid_rows == 3
    assert summary.skipped_rows == 1
    assert summary.rows[0].character_a == 1
    assert summary.rows[0].character_b == 18
    assert summary.rows[0].count == 2
    assert summary.rows[0].by_split == {"train": 1, "val": 1}


def test_summarize_can_preserve_player_order() -> None:
    entries = [
        {
            "players": [{"character": 1}, {"character": 18}],
            "annotation": {"split": "train"},
        },
        {
            "players": [{"character": 18}, {"character": 1}],
            "annotation": {"split": "train"},
        },
    ]

    summary = summarize(entries, ordered=True)

    assert len(summary.rows) == 2
    assert {(row.character_a, row.character_b) for row in summary.rows} == {(1, 18), (18, 1)}


def test_write_tsv(tmp_path: Path) -> None:
    summary = summarize(
        [
            {
                "players": [{"character": 1}, {"character": 1}],
                "annotation": {"split": "test"},
            }
        ]
    )
    path = tmp_path / "matchups.tsv"

    _write_tsv(path, summary)

    assert path.read_text().splitlines() == [
        "rank\tcharacter_a_id\tcharacter_a\tcharacter_b_id\tcharacter_b\tcount\tpercent\ttest",
        "1\t1\tFOX\t1\tFOX\t1\t100.0000\t1",
    ]
