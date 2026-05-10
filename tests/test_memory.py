"""Tests for the SQLite-backed three-tier long-term memory."""

from __future__ import annotations

from memory import LongTermMemory


def test_round_trip_semantic_facts() -> None:
    mem = LongTermMemory()
    fid = mem.remember_fact("user1", "dislike", "okra", source="user_stated")
    assert fid > 0
    facts = mem.recall_facts("user1")
    assert len(facts) == 1
    assert facts[0]["content"] == "okra"
    assert facts[0]["fact_type"] == "dislike"


def test_filter_facts_by_type_and_substring() -> None:
    mem = LongTermMemory()
    mem.remember_fact("u1", "dislike", "okra")
    mem.remember_fact("u1", "dislike", "kale")
    mem.remember_fact("u1", "preference", "high-protein")
    mem.remember_fact("u2", "dislike", "okra")  # different user

    dislikes = mem.recall_facts("u1", fact_type="dislike")
    assert {f["content"] for f in dislikes} == {"okra", "kale"}

    prefs = mem.recall_facts("u1", fact_type="preference")
    assert prefs[0]["content"] == "high-protein"

    okra_only = mem.recall_facts("u1", contains="okra")
    assert len(okra_only) == 1


def test_user_isolation() -> None:
    mem = LongTermMemory()
    mem.remember_fact("alice", "allergy", "peanut")
    mem.remember_fact("bob", "allergy", "shellfish")

    assert {f["content"] for f in mem.recall_facts("alice")} == {"peanut"}
    assert {f["content"] for f in mem.recall_facts("bob")} == {"shellfish"}


def test_forget_fact() -> None:
    mem = LongTermMemory()
    fid = mem.remember_fact("u", "dislike", "okra")
    mem.forget_fact(fid)
    assert mem.recall_facts("u") == []


def test_procedural_records_round_trip() -> None:
    mem = LongTermMemory()
    issues = [{"code": "calorie_deviation", "description": "x", "severity": "medium"}]
    mem.remember_validation("u1", "1500 kcal plan", "revise", issues)

    history = mem.recall_validations("u1")
    assert len(history) == 1
    assert history[0]["verdict"] == "revise"
    assert history[0]["issues"] == issues


def test_episodic_session_round_trip() -> None:
    mem = LongTermMemory()
    payload = {"messages": [{"role": "user", "content": "hi"}], "memory": {"x": 1}}
    mem.remember_session("u1", "session-A", payload)

    sessions = mem.recall_sessions("u1")
    assert len(sessions) == 1
    assert sessions[0]["payload"] == payload
    assert sessions[0]["session_id"] == "session-A"


def test_recall_limit_and_order() -> None:
    mem = LongTermMemory()
    for i in range(15):
        mem.remember_fact("u", "note", f"fact-{i}")
    facts = mem.recall_facts("u", limit=5)
    assert len(facts) == 5
    # newest first
    assert facts[0]["content"] == "fact-14"


def test_knowledge_agent_returns_citations() -> None:
    """KnowledgeAgent should always return a citations list (possibly empty)."""
    from knowledge import KnowledgeAgent

    class StubWebSearch:
        def handle_task(self, query: str) -> str:
            return (
                "Calories per 100g of chicken breast: 165 kcal (USDA FDC).\n"
                "Source: https://fdc.nal.usda.gov/food-details/171477/nutrients"
            )

    agent = KnowledgeAgent(StubWebSearch())
    import json
    raw = agent.handle_task('{"kind": "nutrition", "query": "chicken breast"}', memory={})
    payload = json.loads(raw)
    assert payload["kind"] == "nutrition"
    assert payload["citations"], "Should extract at least one URL"
    assert any("fdc.nal.usda.gov" in c for c in payload["citations"])


def test_knowledge_agent_marks_uncited_answers() -> None:
    """When no citations are found, agent appends an advisory note."""
    from knowledge import KnowledgeAgent
    import json

    class NoCitationStub:
        def handle_task(self, query: str) -> str:
            return "Generic answer without any URL."

    agent = KnowledgeAgent(NoCitationStub())
    payload = json.loads(agent.handle_task("how many calories in an apple?", memory={}))
    assert payload["citations"] == []
    assert "advisory only" in payload["answer"]
