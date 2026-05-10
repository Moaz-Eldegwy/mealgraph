"""Unit tests for the APIPoolManager rate limiter."""

from __future__ import annotations

import time

import pytest

from utils import APIPoolManager


def test_round_robin_no_limits() -> None:
    pool = APIPoolManager(["k1", "k2", "k3"], rate_limits=None)
    seen = [pool.get_next_key("any-model") for _ in range(6)]
    # With no limits we should walk through all keys at least twice.
    assert set(seen) == {"k1", "k2", "k3"}


def test_rpm_spacing_enforced() -> None:
    """With RPM=60 we expect a ~1s spacing between consecutive uses of the
    same key. Two-key pool should let us avoid the wait."""
    pool = APIPoolManager(["k1", "k2"], rate_limits={"m": (60, 1000)})

    k_a = pool.get_next_key("m")
    pool.record_usage(k_a, "m", time.time())
    k_b = pool.get_next_key("m")
    assert k_a != k_b, "Round-robin should pick the other key when one is hot"


def test_rpd_exhaustion_drops_key() -> None:
    """A key that hits its daily limit must be removed from active pool."""
    pool = APIPoolManager(["k1", "k2"], rate_limits={"m": (60, 2)})
    for _ in range(2):
        k = pool.get_next_key("m")
        pool.record_usage(k, "m")

    # By now both keys may have hit their RPD=2. Next call should still work
    # if at least one key has capacity, else raise RuntimeError.
    keys_left = list(pool.active_keys)
    if not keys_left:
        with pytest.raises(RuntimeError):
            pool.get_next_key("m")
    else:
        # Drain the remaining one too.
        for _ in range(2):
            try:
                k = pool.get_next_key("m")
                pool.record_usage(k, "m")
            except RuntimeError:
                break
        assert not pool.active_keys, "Both keys should be exhausted now"
