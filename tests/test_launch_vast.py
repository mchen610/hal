import runpy
from pathlib import Path

LAUNCH_VAST = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "launch_vast.py"))


def test_queue_forwards_offer_filters(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen: dict[str, object] = {}

    def fake_search(_vast, **kwargs):  # type: ignore[no-untyped-def]
        seen["min_inet_down"] = kwargs["min_inet_down"]
        seen["exclude_machine_ids"] = kwargs["exclude_machine_ids"]
        return [{"id": 1}]

    monkeypatch.setitem(LAUNCH_VAST["queue"].__globals__, "search", fake_search)
    offers = LAUNCH_VAST["queue"](
        object(),
        max_price=0.75,
        limit=1,
        disk=500,
        min_vram=0,
        min_ram=64,
        min_dlperf=150,
        min_inet_down=700,
        exclude_machine_ids=[46319],
        data_gb=40,
        upload_gb=1,
        run_hours=2,
        poll_interval_s=0,
    )

    assert offers == [{"id": 1}]
    assert seen["min_inet_down"] == 700
    assert seen["exclude_machine_ids"] == [46319]


def test_build_query_excludes_machine_ids() -> None:
    query = LAUNCH_VAST["build_query"](0.75, 500, 0, 64, 150, 700, [46319, 12345])

    assert "machine_id notin [46319,12345]" in query
