"""Pure helpers for per-item fan-out: item keys, slicing one input into items, and aggregating item records.

An item's text is always a verbatim slice of the input it came from; its key is derived from that text, so splitting
the same input again yields the same keys (and the same per-item cache entries).
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Mapping

from .guardrails import is_cjk_token, quote_tokens, substantive

MAX_ITEMS = 50
DEFAULT_MAX_ITEMS = 20
ITEM_KEY = re.compile(r"it_[0-9a-f]{10}")  # used with fullmatch

_WORD_BEFORE = r"(?<![^\W_])"
_WORD_AFTER = r"(?![^\W_])"


def item_key(text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    return "it_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]


def split_by_separator(packet: str, separator: str) -> list[str]:
    return [piece.strip() for piece in packet.split(separator) if piece.strip()]


def _quote_pattern(quote: str) -> re.Pattern | None:
    """The quote's tokens in order as whole tokens, anything that is not a letter or digit between them,
    case-insensitive (quotes copied from text routinely drop or move commas and quote marks). None when the quote
    has no tokens."""
    parts = []
    for token in quote_tokens(quote):
        escaped = re.escape(token)
        parts.append(escaped if is_cjk_token(token) else _WORD_BEFORE + escaped + _WORD_AFTER)
    return re.compile(r"[\W_]*".join(parts), re.IGNORECASE) if parts else None


def slice_by_boundaries(packet: str, starts: list[str]) -> tuple[list[str], list[str]]:
    """Cut `packet` at each start quote, in order. Returns (texts, problems).

    Each start is searched forward from the previous item's start and must be substantive, found, and unique in the
    rest of the packet. Item n runs from start n to start n+1 (the last one to the end); text before the first start
    is ignored. Problems read "start_quote <n> not found|ambiguous|too short" (n counts from 1)."""
    offsets: list[int] = []
    problems: list[str] = []
    position = 0
    for n, start in enumerate(starts, 1):
        pattern = _quote_pattern(start) if isinstance(start, str) and substantive(start) else None
        if pattern is None:
            problems.append(f"start_quote {n} too short")
            continue
        matches = list(pattern.finditer(packet, position))
        if not matches:
            problems.append(f"start_quote {n} not found")
            continue
        if len(matches) > 1:
            problems.append(f"start_quote {n} ambiguous")
            continue
        offsets.append(matches[0].start())
        position = matches[0].end()
    ends = offsets[1:] + [len(packet)]
    return [packet[a:b] for a, b in zip(offsets, ends)], problems


def build_items(labels: list[str], texts: list[str]) -> tuple[list[dict], list[str]]:
    """[{key, label, text}] in order. Items whose text matches an earlier item's are dropped; their labels are
    returned as duplicates."""
    items: list[dict] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for label, text in zip(labels, texts):
        key = item_key(text)
        if key in seen:
            duplicates.append(label)
            continue
        seen.add(key)
        items.append({"key": key, "label": label, "text": text})
    return items, duplicates


def active_items(roster: Iterable[Mapping]) -> list[Mapping]:
    return [r for r in roster if r.get("status") == "active"]


def aggregate(roster: list[Mapping], records: list[Mapping]) -> dict:
    """What later steps read from a per-item step: active items in roster order and the excluded ones."""
    by_key = {r["key"]: r for r in records}
    items = [{"key": r["key"], "label": r["label"], "state": by_key[r["key"]]["state"],
              "output": by_key[r["key"]].get("output")}
             for r in active_items(roster) if r["key"] in by_key]
    excluded = [{"key": r["key"], "label": r["label"], "reason": r.get("reason")}
                for r in roster if r.get("status") == "excluded"]
    return {"items": items, "excluded": excluded}


def aggregate_state(records: list[Mapping]) -> tuple[str, dict | None]:
    """The step state from its active item records: completed when all completed; else waiting_approval when any
    waits for a review; else needs_attention naming the items that need it; else "pending" (items not run yet, so
    the caller names the cap that stopped them)."""
    if not records:
        return "needs_attention", {"code": "no_items_left"}
    states = [r["state"] for r in records]
    if all(s == "completed" for s in states):
        return "completed", None
    if "waiting_review" in states:
        return "waiting_approval", None
    attention = [r["key"] for r in records if r["state"] in ("needs_attention", "failed")]
    if attention:
        return "needs_attention", {"code": "items_need_attention", "items": attention}
    return "pending", None
