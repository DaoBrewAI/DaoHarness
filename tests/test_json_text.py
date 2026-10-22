"""Finding the JSON object in a model's reply: what is tidied and reported, and what stays an error."""
import json
import random

import pytest

from daoharness.json_text import JsonTextError, read_json_object

THINK = "removed a <think> block"
FENCE = "removed a code fence"
COMMA = "removed a trailing comma"
BEFORE = "ignored text before the JSON object"
AFTER = "ignored text after the JSON object"


def _tidy(reply, **kw):
    value, repairs = read_json_object(reply, **kw)
    return value, repairs


def _reason(reply, **kw):
    with pytest.raises(JsonTextError) as raised:
        read_json_object(reply, **kw)
    return raised.value.reason


# --- nothing to tidy ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reply", ['{"a": 1}', '  \n{"a": 1}\n  ', '{"a": {"b": [1, 2, {"c": null}]}}'])
def test_a_plain_object_is_read_as_it_is(reply):
    value, repairs = _tidy(reply)
    assert isinstance(value, dict) and repairs == []


def test_text_inside_the_object_is_never_touched():
    reply = ('{"a": "use <think>x</think> tags", "b": "```json", "c": "x,}", "d": "}{", "e": "say \\"}\\"", '
             '"f": "ends with a comma,"}')
    value, repairs = _tidy(reply)
    assert value["a"] == "use <think>x</think> tags" and value["b"] == "```json" and value["c"] == "x,}"
    assert value["d"] == "}{" and value["e"] == 'say "}"' and repairs == []


# --- what is tidied, and said ------------------------------------------------------------------------------------

@pytest.mark.parametrize("reply, repairs", [
    ('<think>a {b} c</think>\n{"a": 1}', [THINK]),
    ('<think>one {</think><think>two }</think>{"a": 1}', ["removed 2 <think> blocks"]),
    ('<THINK>loud</THINK>{"a": 1}', [THINK]),
    ('<thinking>hmm {x}</thinking>{"a": 1}', [THINK]),
    ('{"a": 1}<think>afterwards</think>', [THINK]),
    ('I thought about {it}.\n</think>\n{"a": 1}', ["removed the reasoning that ends at </think>"]),
    ('```json\n{"a": 1}\n```', [FENCE]),
    ('```\n{"a": 1}\n```', [FENCE]),
    ('```JSON\n{"a": 1}\n```', [FENCE]),
    ('~~~json\n{"a": 1}\n~~~', [FENCE]),
    ('```json {"a": 1} ```', [FENCE]),
    ('```json\n{"a": 1}', [FENCE]),
    ('{"a": 1}\n```', [FENCE]),
    ('```json\n```json\n{"a": 1}\n```\n```', [FENCE]),
    ('Sure! {"a": {"b": 2}} Done.', [BEFORE, AFTER]),
    ('Here it is:\n```json\n{"a": 1}\n```', [FENCE, BEFORE]),
    ('{"a": 1,}', [COMMA]),
    ('{"a": [1, 2,], "b": {"c": 3,},}', ["removed 3 trailing commas"]),
    ('{"a": [1,\n  2 , \n ]\n}', [COMMA]),
    ('<think>{x}</think>\n```json\n{"a": [1,],}\n```\nDone!', [THINK, FENCE, AFTER, "removed 2 trailing commas"]),
])
def test_wrapping_is_removed_and_each_removal_is_reported(reply, repairs):
    value, found = _tidy(reply)
    assert value["a"] is not None and found == repairs


def test_the_object_is_the_same_object_after_tidying():
    assert _tidy('<think>{"a": 0}</think>\n```json\n{"a": [1, 2,], "b": "x,",}\n```')[0] == {"a": [1, 2], "b": "x,"}


# --- what stays an error -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("reply, reason", [
    ('{"a": 1}\n{"a": 2}', "extra data after the JSON object"),
    ('{"a": 1}{"a": 2}', "extra data after the JSON object"),
    ('{"a": 1}, {"a": 2}', "extra data after the JSON object"),
    ('```json\n{"a": 1}\n```\n```json\n{"a": 2}\n```', "extra data after the JSON object"),
    ('{"a": 1} [1, 2]', "extra data after the JSON object"),
    ('{"a": 1}}', "extra data after the JSON object"),
    ('{"a": 1} then {braces}', "extra data after the JSON object"),
    ('{"a": 1}\nSee [the docs] for more.', "extra data after the JSON object"),
    ('[{"a": 1}]', "extra data before the JSON object"),
    ('Use {braces}: {"a": 1}', "extra data after the JSON object"),   # only the first brace can start the object
    ('{"a": 1}]', "extra data after the JSON object"),
])
def test_a_second_value_or_a_list_around_the_object_is_refused(reply, reason):
    assert reason in _reason(reply)


@pytest.mark.parametrize("reply", ['{"a": 1}\n{"a": 2}', '[{"a": 1}]', '{"a": 1} [1, 2]'])
def test_the_error_does_not_return_a_cut_down_object(reply):
    with pytest.raises(JsonTextError):
        read_json_object(reply)


@pytest.mark.parametrize("reply, reason", [
    ("", "no JSON object"),
    ("I cannot do that.", "no JSON object"),
    ("[1, 2]", "no JSON object"),
    ('"just a string"', "no JSON object"),
    ('<think>only thinking</think>', "no JSON object"),
    ('{"a": [1, 2', "not finished"),
    ('{"a": "cut off', "not finished"),
    ('<think>{"a": 1}', "<think> block"),
    ("{'a': 1}", "syntax error"),
    ("{a: 1}", "syntax error"),
    ('{"a": 1 // one\n}', "syntax error"),
    ('{"a": 1 "b": 2}', "syntax error"),
    ('{"a": [,1]}', "syntax error"),
    ('{"a": [1,,2]}', "syntax error"),
    ('{,}', "syntax error"),
    ('{"a": 1,, }', "syntax error"),
    ('{"a": NaN}', "syntax error"),
    ('{"a": 01}', "syntax error"),
])
def test_text_that_is_not_strict_json_stays_an_error(reply, reason):
    assert reason in _reason(reply)


def test_a_syntax_error_says_where():
    reason = _reason('{\n  "a": 1\n  "b": 2\n}')
    assert "line 3, column 3" in reason


def test_nan_is_not_json():
    # json.loads accepts NaN and Infinity; a reply is JSON, and JSON has neither
    assert "syntax error" in _reason('{"a": Infinity}')


# --- text that is not an object is not read for one --------------------------------------------------------------

def test_prose_is_not_allowed_around_a_nested_value_when_wrapping_is_off():
    assert _tidy('```json\n{"a": 1,}\n```', wrapped=False) == ({"a": 1}, [FENCE, COMMA])
    assert "extra data before" in _reason('Here: {"a": 1}', wrapped=False)
    assert "extra data after" in _reason('{"a": 1} Done.', wrapped=False)
    assert "extra data before" in _reason('<think>x</think>{"a": 1}', wrapped=False)


# --- a closing <think> tag that nothing opened ---------------------------------------------------------------------

@pytest.mark.parametrize("reply, repairs", [
    ('{"z</think>": null}', []),
    ('```json\n{"z</think>": null}\n```', [FENCE]),
    ('Here it is: {"a": "</think>"}', [BEFORE]),
])
def test_a_closing_tag_that_is_only_text_inside_the_object_is_left_in_it(reply, repairs):
    value, found = _tidy(reply)
    assert found == repairs and "</think>" in json.dumps(value)


def test_reasoning_that_ends_at_a_closing_tag_is_dropped_and_the_answer_read_whole():
    value, repairs = _tidy('thoughts with {braces} and {"draft": 1}.\n</think>\n{"z</think>": null}')
    assert value == {"z</think>": None} and repairs == ["removed the reasoning that ends at </think>"]


@pytest.mark.parametrize("reply", ['{"a": "x</think>", "b": {}', 'Here: {"a</think>": {}', '```json\n{"a</think>": {}'])
def test_an_object_that_stops_short_with_the_tag_inside_is_not_guessed_at(reply):
    with pytest.raises(JsonTextError) as raised:
        read_json_object(reply)
    assert raised.value.cut_off and "not finished" in raised.value.reason


def test_a_reply_that_starts_with_an_object_is_not_reasoning():
    assert "extra data after" in _reason('{"a": 1} </think> {"b": 2}')


def test_reasoning_with_no_answer_after_it_is_an_error():
    assert "no JSON object" in _reason('I could answer {"a": 1} here.\n</think>\n')


# --- any object, wrapped or cut --------------------------------------------------------------------------------------

def _random_object(rng):
    strings = ["s", "a,b", "x}y", "q]", "<think>t</think>", "</think>", "```", "\\\"", "", "\u65e5\u672c"]

    def value(depth=0):
        kind = rng.random()
        if depth > 2 or kind < 0.4:
            return rng.choice(strings + [7, 1.5, True, None])
        if kind < 0.7:
            return [value(depth + 1) for _ in range(rng.randint(0, 3))]
        return {rng.choice(strings): value(depth + 1) for _ in range(rng.randint(0, 3))}

    return {rng.choice(strings) or "k": value() for _ in range(rng.randint(0, 4))}


def test_random_objects_come_back_whole_however_they_are_wrapped_and_never_from_a_cut_or_a_pair():
    rng = random.Random(20261022)
    wraps = [lambda s: s, lambda s: f"```json\n{s}\n```", lambda s: f"<think>a {{b}} c</think>\n{s}",
             lambda s: f"Sure!\n{s}\nDone.", lambda s: f"thoughts {{x}}\n</think>\n```json\n{s}\n```"]
    for _ in range(300):
        wanted = _random_object(rng)
        text = json.dumps(wanted, indent=rng.choice([None, 2]))
        assert read_json_object(rng.choice(wraps)(text))[0] == wanted
        for extra in ('{"x": 1}', "[1]", "}"):
            with pytest.raises(JsonTextError):
                read_json_object(text + "\n" + extra)
        with pytest.raises(JsonTextError):
            read_json_object(text[:rng.randint(1, len(text) - 1)])
