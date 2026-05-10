"""Run the offline eval harness from pytest so CI catches regressions."""

from __future__ import annotations

from evals.runner import run_offline


def test_offline_eval_all_fixtures_pass() -> None:
    results = run_offline()
    assert results, "Eval harness returned no results"
    failed = [r for r in results if not r.passed]
    assert not failed, "Failures:\n" + "\n".join(
        f"  {r.name}: {r.failures}" for r in failed
    )


def test_observability_metrics_snapshot() -> None:
    from observability import get_metrics, span

    get_metrics().reset()
    with span("test_span", kind="agent"):
        pass
    snap = get_metrics().snapshot()
    assert "test_span" in snap["agents"]
    assert snap["agents"]["test_span"]["calls"] == 1


def test_init_langsmith_off_by_default(monkeypatch) -> None:
    import os
    from observability import init_langsmith

    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert init_langsmith() is False
