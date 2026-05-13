"""KnowledgeAgent — citation-first retrieval over authoritative sources.

The default implementation is backed by :class:`tools.WebSearchTool` and
biases queries toward authoritative domains (USDA / WHO / ADA / EFSA /
NICE) per kind. The interface is intentionally minimal so a RAG-backed
variant over an embedded USDA + clinical-guideline index can drop in
without touching the agents that call it.

Contract: every response is a synthesised answer with at least one
citation when the source supplies one. The Validator flags medical
recommendations that arrive without citations, so this agent is the
provenance layer for clinical content.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from logging_setup import get_logger
from utils import save_to_json

_logger = get_logger("agents.knowledge")


class KnowledgeAgent:
    """Default Knowledge implementation backed by WebSearchTool.

    Future drop-in: replace ``self.web`` with a RAG retriever that walks an
    embedded USDA/WHO/ADA/EFSA index and returns citation tuples.
    """

    SUPPORTED_KINDS = {"nutrition", "guideline", "drug_interaction", "general"}

    def __init__(self, web_search_tool, llm_instance: Optional[Any] = None) -> None:
        self.web = web_search_tool
        self.llm = llm_instance  # reserved for the RAG-backed variant

    def handle_task(self, task: str, memory: Dict[str, Any]) -> str:  # noqa: ARG002
        """Answer ``task`` and return JSON ``{answer, citations}``.

        ``task`` is a free-text question. Optional structured form:
            ``{"kind": "nutrition" | "guideline" | ...,
               "query": "...",
               "context": "..."}``
        """
        kind, query, context = self._parse_task(task)
        _logger.info("📚 KNOWLEDGE: kind=%s query=%r", kind, query[:80])

        # Bias the query toward citation-rich sources.
        biased_query = self._bias_query(kind, query)
        web_answer = self.web.handle_task(biased_query)

        citations = self._extract_citations(web_answer)
        answer = web_answer  # WebSearch already synthesises; we just append a citation note.
        if not citations:
            answer += (
                "\n\n[Note] This answer comes from a generalist web search; "
                "no authoritative clinical citation was found. Treat as advisory only."
            )

        payload = {"kind": kind, "answer": answer, "citations": citations}
        save_to_json(
            {
                "task": task,
                "kind": kind,
                "query": query,
                "context": context,
                "biased_query": biased_query,
                "answer": answer,
                "citations": citations,
                "timestamp": datetime.now().isoformat(),
            },
            f"knowledge_{datetime.now().isoformat()}.json",
            subdirectory="KnowledgeAgent",
        )
        return json.dumps(payload)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_task(task: str) -> tuple[str, str, str]:
        try:
            data = json.loads(task)
            if isinstance(data, dict):
                kind = data.get("kind", "general")
                if kind not in KnowledgeAgent.SUPPORTED_KINDS:
                    kind = "general"
                return kind, data.get("query", ""), data.get("context", "")
        except (json.JSONDecodeError, TypeError):
            pass
        return "general", task, ""

    @staticmethod
    def _bias_query(kind: str, query: str) -> str:
        """Steer the search toward authoritative domains per kind."""
        if kind == "nutrition":
            return (
                f"{query} site:fdc.nal.usda.gov OR site:nutritionsource.hsph.harvard.edu "
                "OR site:who.int"
            )
        if kind == "guideline":
            return (
                f"{query} site:who.int OR site:diabetes.org OR site:efsa.europa.eu "
                "OR site:nice.org.uk"
            )
        if kind == "drug_interaction":
            return f"{query} site:medlineplus.gov OR site:fda.gov OR site:nih.gov"
        return query

    @staticmethod
    def _extract_citations(text: str) -> List[str]:
        """Pull URL-looking tokens out of the synthesised answer."""
        import re

        urls = re.findall(r"https?://[^\s)\]]+", text)
        # De-duplicate while preserving order.
        seen = set()
        out: List[str] = []
        for u in urls:
            u_clean = u.rstrip(".,);")
            if u_clean not in seen:
                seen.add(u_clean)
                out.append(u_clean)
        return out


__all__ = ["KnowledgeAgent"]
