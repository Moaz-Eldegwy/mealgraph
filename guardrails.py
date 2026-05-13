"""Input/output guardrails.

* :func:`detect_prompt_injection` — keyword/regex sniff for the common
  jailbreak patterns (ignore previous, system override, role-play override).
  Returns ``(is_attempt, matched_patterns)`` rather than auto-blocking, so
  the caller decides the response (refuse, log, escalate, …).

* :func:`redact_pii` — masks email, phone, SSN-shaped numbers, lab IDs,
  and obvious medical-record numbers in any text destined for logs and
  traces. Working memory keeps the original strings so the plan can still
  address the user by name; only the persisted log layer redacts.

* :func:`hitl_chip` — formats the human-in-the-loop escalation marker the
  Gradio app renders as a coloured badge.

These checks are intentionally lightweight — a production deployment
would layer a small classifier (NeMo Guardrails, Lakera, …) on top.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Prompt-injection detector
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("ignore_previous", re.compile(r"\bignore\s+(all\s+)?previous\s+(instructions?|prompts?|rules?)\b", re.I)),
    ("system_override", re.compile(r"\b(system\s+(prompt|message)|developer\s+message)\b", re.I)),
    ("role_override", re.compile(r"\byou\s+are\s+now\s+(?:a|an)\s+\w+", re.I)),
    ("jailbreak_persona", re.compile(r"\b(DAN|do\s+anything\s+now|jailbreak|sudo\s+mode)\b", re.I)),
    ("tool_misuse", re.compile(r"\b(execute|run|eval)\s+(arbitrary|the\s+following)\s+code\b", re.I)),
    ("fence_smuggle", re.compile(r"```(?:json|python)?\s*\n.*?(system|assistant)\s*[:=]", re.I | re.S)),
    ("data_exfil", re.compile(r"\b(print|reveal|leak|exfiltrate)\s+(your|the)\s+(system\s+)?prompt\b", re.I)),
]


@dataclass(frozen=True)
class InjectionVerdict:
    is_attempt: bool
    matches: List[str]

    @property
    def severity(self) -> str:
        if not self.is_attempt:
            return "none"
        if len(self.matches) >= 2:
            return "high"
        return "medium"


def detect_prompt_injection(text: str) -> InjectionVerdict:
    """Run all patterns; return list of named matches (empty if clean)."""
    if not text:
        return InjectionVerdict(False, [])
    matches = [name for name, rx in _INJECTION_PATTERNS if rx.search(text)]
    return InjectionVerdict(bool(matches), matches)


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------
_PII_PATTERNS: List[Tuple[str, re.Pattern[str], str]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    # Common phone shapes — international, US, dotted, dashed
    (
        "PHONE",
        re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?){2}\d{4}\b"),
        "[REDACTED_PHONE]",
    ),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    # 6-12 digit medical record number with MRN/EHR context word
    (
        "MRN",
        re.compile(r"\b(?:MRN|EHR|medical\s+record(?:\s+number)?)[\s:#]*\d{4,12}\b", re.I),
        "[REDACTED_MRN]",
    ),
    # Credit-card-like 13-19 digit runs (with optional separators)
    (
        "CC",
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        "[REDACTED_CC]",
    ),
]


def redact_pii(text: str) -> str:
    """Return ``text`` with detected PII tokens replaced by placeholders."""
    if not text:
        return text
    out = text
    for _name, rx, replacement in _PII_PATTERNS:
        out = rx.sub(replacement, out)
    return out


# ---------------------------------------------------------------------------
# Human-in-the-loop escalation marker
# ---------------------------------------------------------------------------
HITL_MARKER = "<<HITL:CLINICIAN_REVIEW_REQUIRED>>"


def hitl_chip(reason: str) -> str:
    """Wrap ``reason`` in a stable marker the UI/parsers can detect."""
    safe_reason = reason.replace("\n", " ").strip()
    return f"{HITL_MARKER} {safe_reason}"


def has_hitl_chip(text: str) -> bool:
    return HITL_MARKER in (text or "")


__all__ = [
    "InjectionVerdict",
    "HITL_MARKER",
    "detect_prompt_injection",
    "redact_pii",
    "hitl_chip",
    "has_hitl_chip",
]
