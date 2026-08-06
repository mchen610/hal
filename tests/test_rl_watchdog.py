import queue
import threading
import time

import pytest
from melee_collector import CollectStats
from melee_train import _check_run_health
from melee_train import _next_iteration
from rl_config import WatchdogConfig


def _counters() -> dict[str, int]:
    return {"low_live_iters": 0, "low_sps_iters": 0}


def test_next_iteration_times_out_when_collector_is_alive_but_stalled() -> None:
    stop = threading.Event()

    def idle_collector() -> None:
        while not stop.is_set():
            time.sleep(0.01)

    collector = threading.Thread(target=idle_collector)
    collector.start()
    try:
        with pytest.raises(TimeoutError, match="no rollout iteration"):
            _next_iteration(queue.Queue(), collector, stop, max_wait_s=0.01)
        assert stop.is_set()
    finally:
        stop.set()
        collector.join(timeout=1.0)


def test_run_health_fails_after_consecutive_low_live_boot_iterations() -> None:
    counters = _counters()
    watchdog = WatchdogConfig(min_live_boot_fraction=0.75, max_low_live_iters=2, min_lockstep_sps=0)
    stats = CollectStats(frames=1, lockstep_sps=50.0, live_boots=2, total_boots=4)

    _check_run_health(watchdog, counters, stats)
    with pytest.raises(RuntimeError, match="live boots"):
        _check_run_health(watchdog, counters, stats)


def test_run_health_fails_after_consecutive_low_sps_iterations() -> None:
    counters = _counters()
    watchdog = WatchdogConfig(min_live_boot_fraction=0, min_lockstep_sps=10.0, max_low_sps_iters=2)
    stats = CollectStats(frames=1, lockstep_sps=5.0, live_boots=4, total_boots=4)

    _check_run_health(watchdog, counters, stats)
    with pytest.raises(RuntimeError, match="lockstep_sps"):
        _check_run_health(watchdog, counters, stats)
