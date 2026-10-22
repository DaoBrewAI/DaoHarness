"""Finding the JSON object in text a model wrote.

Models wrap their JSON in thinking, a code fence or a sentence, and leave a comma before a closing bracket. Those are
taken off, and read_json_object says what it took off. Nothing else is guessed at: JSON that is not strict JSON stays
an error, and so does a second value, a list or more braces around the object, because dropping them would drop part
of the reply.
"""
from __future__ import annotations

import json
import re

THINK_TAG = re.compile(r"</?(?:think|thinking|reasoning)\s*>", re.I)
FENCE = re.compile(r"(?:`{3,}|~{3,})[A-Za-z0-9_+.-]*")   # a code fence, with the language glued to it
STRUCTURE = re.compile(r"[{}\[\]]")


class JsonTextError(ValueError):
    """The text holds no JSON object that can be read. `reason` says why; `cut_off` is true when an object starts
    and the text ends before it does."""

    def __init__(self, reason: str, cut_off: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cut_off = cut_off


def read_json_object(text: str, *, wrapped: bool = True) -> tuple[dict, list[str]]:
    """(the JSON object in `text`, what was taken off to get it), or JsonTextError.

    Taken off: <think> blocks, code fences, a sentence before or after the object, and a trailing comma. With
    `wrapped` False (a JSON object written inside a string) only code fences and trailing commas are. The line and
    column in a reason count from the start of the object."""
    if wrapped:
        tag = THINK_TAG.search(text)
        if tag is not None and tag.group(0).startswith("</") and not FENCE.sub("", text).lstrip().startswith("{"):
            return _after_reasoning(text, tag)
    value, repairs, _, _ = _read(text, wrapped)
    return value, repairs


def _after_reasoning(text: str, closer: re.Match) -> tuple[dict, list[str]]:
    """The first <think> tag in the reply closes a block nothing opened (the opening tag was part of the prompt): the
    reasoning ends there and the answer follows. Where the tag is only text inside an object, the object is read as it
    stands, and where it may be text inside an object that stops short, nothing is guessed."""
    try:
        value, repairs, start, end = _read(text, True)
    except JsonTextError as exc:
        if exc.cut_off:
            raise
    else:
        if start < closer.start() < end:
            return value, repairs
    value, repairs, _, _ = _read(text[closer.end():], True)
    return value, ["removed the reasoning that ends at </think>", *repairs]


def _read(text: str, wrapped: bool) -> tuple[dict, list[str], int, int]:
    """(the object, what was taken off, where the object starts and ends in `text`)."""
    repairs: list[str] = []
    spans, unclosed = _thinking(text) if wrapped else ([], None)
    start = _first_brace(text, spans)
    if start < 0:
        raise JsonTextError("no JSON object found in the reply")
    if unclosed is not None and unclosed < start:
        raise JsonTextError("the reply stops inside a <think> block, before any answer")
    end = _object_end(text, start)
    if end is None:
        raise JsonTextError("the JSON object is not finished", cut_off=True)
    object_text = text[start:end]

    sides = {"before": text[:start], "after": text[end:]}
    blocks = 0
    for side, part in sides.items():
        if wrapped:
            sides[side], count = _without_blocks(part)
            blocks += count
    if blocks:
        repairs.append("removed a <think> block" if blocks == 1 else f"removed {blocks} <think> blocks")
    if any(FENCE.search(part) for part in sides.values()):
        repairs.append("removed a code fence")
    for side, part in sides.items():
        part = FENCE.sub("", part)
        if STRUCTURE.search(part) or (not wrapped and part.strip()):
            raise JsonTextError(f"extra data {side} the JSON object")
        if part.strip():
            repairs.append(f"ignored text {side} the JSON object")

    value, commas = _load(object_text)
    if commas:
        repairs.append("removed a trailing comma" if commas == 1 else f"removed {commas} trailing commas")
    return value, repairs, start, end


def _load(object_text: str) -> tuple[dict, int]:
    """(the object, the trailing commas removed to read it). Strict JSON otherwise: no NaN, no comments."""
    try:
        return _loads(object_text), 0
    except (ValueError, RecursionError) as first:
        fixed, commas = _drop_trailing_commas(object_text)
        if not commas:
            raise JsonTextError(_syntax(first)) from None
    try:
        return _loads(fixed), commas
    except (ValueError, RecursionError) as second:
        raise JsonTextError(_syntax(second)) from None


def _loads(text: str) -> dict:
    return json.loads(text, parse_constant=_not_json)


def _not_json(name: str):
    raise ValueError(f"{name} is not JSON")


def _syntax(exc: BaseException) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"syntax error: {exc.msg} (line {exc.lineno}, column {exc.colno})"
    if isinstance(exc, RecursionError):
        return "syntax error: nested too deeply"
    return f"syntax error: {exc}"


def _thinking(text: str) -> tuple[list[tuple[int, int]], int | None]:
    """(the spans of complete <think> blocks, where a block that is never closed starts). A closing tag that no
    opening tag precedes is left alone."""
    spans: list[tuple[int, int]] = []
    opened = None
    for tag in THINK_TAG.finditer(text):
        if not tag.group(0).startswith("</"):
            opened = tag.start() if opened is None else opened
        elif opened is not None:
            spans.append((opened, tag.end()))
            opened = None
    return spans, opened


def _first_brace(text: str, spans: list[tuple[int, int]]) -> int:
    """The first "{" that is not inside a <think> block, or -1."""
    at = text.find("{")
    while at >= 0:
        inside = next((b for a, b in spans if a <= at < b), None)
        if inside is None:
            return at
        at = text.find("{", inside)
    return -1


def _without_blocks(part: str) -> tuple[str, int]:
    """(the text without its complete <think> blocks, how many there were)."""
    out, last = [], 0
    spans, _ = _thinking(part)
    for a, b in spans:
        out.append(part[last:a])
        last = b
    out.append(part[last:])
    return "".join(out), len(spans)


def _object_end(text: str, start: int) -> int | None:
    """The index just past the bracket that closes the object at `start`, or None when the text ends first. Quotes
    are followed, so a bracket inside a string does not count; a bracket of the wrong kind ends the scan there, for
    the parser to name."""
    closers: list[str] = []
    in_string = escaped = False
    for at in range(start, len(text)):
        char = text[at]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            closers.append("}" if char == "{" else "]")
        elif char in "}]":
            if not closers or closers.pop() != char or not closers:
                return at + 1
    return None


def _drop_trailing_commas(text: str) -> tuple[str, int]:
    """(the text without a comma that only whitespace separates from a closing bracket, how many were dropped).
    Commas inside strings stay, and so does a comma with no value before it: that is not a trailing comma."""
    out: list[str] = []
    dropped = 0
    in_string = escaped = False
    previous = ""   # the last character outside a string that is not whitespace
    for at, char in enumerate(text):
        out.append(char)
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                previous = char
        elif char == '"':
            in_string = True
        elif char == "," and previous not in ("", "{", "[", ",") and _closes_next(text, at + 1):
            out.pop()
            dropped += 1
        elif not char.isspace():
            previous = char
    return "".join(out), dropped


def _closes_next(text: str, at: int) -> bool:
    while at < len(text) and text[at].isspace():
        at += 1
    return at < len(text) and text[at] in "}]"
