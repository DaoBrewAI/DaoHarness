"""Deterministic post-step guardrails. Pure functions; no I/O."""
from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class Check:
    id: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "passed": self.passed, "detail": self.detail}


def flatten_text(value: Any) -> list[str]:
    """Every string leaf in depth-first, left-to-right order. Linear in the size of the value."""
    out: list[str] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, Mapping):
            stack.extend(reversed(list(item.values())))
        elif isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
    return out


_FOLD = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_FOLD)
    return " ".join(text.split()).casefold()


def normalize_text(text: str) -> str:
    """NFKC, quote/dash folding, collapsed whitespace, casefold."""
    return _normalize(text)


# Kana, CJK ideographs (incl. extension A and compatibility) and Hangul syllables: scripts written without spaces,
# so each character is its own token.
_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff"
_CJK_CHAR = re.compile(f"[{_CJK}]")
# Numbers keep their inner punctuation ("3.5" is not "3-5", "1,500" is not "1.500"); then single CJK characters;
# then runs of other letters and digits.
_TOKEN = re.compile(rf"\d+(?:[.,:/\-]\d+)+|[{_CJK}]|[^\W_{_CJK}]+")


def quote_tokens(text: str) -> list[str]:
    """The normalized tokens of a text, in order (the split finds item starts by them)."""
    return _TOKEN.findall(_normalize(text or ""))


def is_cjk_token(token: str) -> bool:
    return bool(_CJK_CHAR.fullmatch(token))


def substantive(quote: str) -> bool:
    """Long enough to mark a place in a text: two words and eight characters, or four CJK characters."""
    tokens = quote_tokens(quote)
    return (len(tokens) >= 2 and len(normalize_text(quote)) >= 8) or sum(map(is_cjk_token, tokens)) >= 4


def values_at(data: Any, path: str) -> list:
    current = [data]
    for part in path.split("."):
        many = part.endswith("[]")
        key = part[:-2] if many else part
        nxt = []
        for node in current:
            if isinstance(node, Mapping) and key in node:
                value = node[key]
                if many:
                    nxt.extend(value if isinstance(value, list) else [])
                else:
                    nxt.append(value)
        current = nxt
    return current


def map_at(data: Any, path: str, fn: Callable[[Any], Any]) -> Any:
    result = copy.deepcopy(data)

    def walk(node, parts):
        if not parts or not isinstance(node, Mapping):
            return
        part, rest = parts[0], parts[1:]
        many = part.endswith("[]")
        key = part[:-2] if many else part
        if key not in node:
            return
        if many and isinstance(node[key], list):
            if not rest:
                node[key] = [fn(v) for v in node[key]]
            else:
                for child in node[key]:
                    walk(child, rest)
        elif not rest:
            node[key] = fn(node[key])
        else:
            walk(node[key], rest)

    walk(result, path.split("."))
    return result


def check_schema(output: Any, schema: Mapping) -> Check:
    errors = sorted(Draft202012Validator(dict(schema)).iter_errors(output), key=lambda e: list(e.absolute_path))
    if not errors:
        return Check("schema", True)
    first = errors[0]
    where = "/".join(str(p) for p in first.absolute_path) or "(root)"
    return Check("schema", False, f"{where}: {first.message[:300]}")


def check_required(output: Any, paths: Iterable[str]) -> Check:
    empty = [p for p in paths
             if not values_at(output, p) or any(not isinstance(v, str) or not v.strip() for v in values_at(output, p))]
    return Check("required_fields", not empty, ", ".join(empty))


# Neutral numeric shapes; order matters (most specific first).
_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{2}-\d{7}\b"),
    re.compile(r"\b\d{4}(?:[ -]\d{4}){2,3}\b"),
    re.compile(r"\b\d{8,19}\b"),
)


def mask_text(value: Any) -> tuple[Any, int]:
    """Mask every string leaf of value (strings, mappings, lists, tuples). Returns (masked copy, count)."""
    if isinstance(value, Mapping):
        total, masked = 0, {}
        for key, child in value.items():
            masked[key], n = mask_text(child)
            total += n
        return masked, total
    if isinstance(value, (list, tuple)):
        total, items = 0, []
        for child in value:
            masked_child, n = mask_text(child)
            items.append(masked_child)
            total += n
        return (tuple(items) if isinstance(value, tuple) else items), total
    if not isinstance(value, str):
        return value, 0
    count = 0

    def repl(match):
        nonlocal count
        count += 1
        return "•••• " + re.sub(r"\D", "", match.group(0))[-4:]

    for pattern in _PATTERNS:
        value = pattern.sub(repl, value)
    return value, count


def mask_sensitive(output: Any, paths: Iterable[str]) -> tuple[Any, Check]:
    total = 0

    def fn(value):
        nonlocal total
        masked, n = mask_text(value)
        total += n
        return masked

    for path in paths:
        output = map_at(output, path, fn)
    return output, Check("mask_sensitive", True, f"masked {total}")
