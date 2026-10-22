"""Pure helpers for per-item fan-out: item keys, packet slicing and aggregation."""
from daoharness.items import (
    ITEM_KEY, aggregate, aggregate_state, build_items, item_key, slice_by_boundaries, split_by_separator)


def test_item_key_is_stable_under_whitespace_and_case():
    key = item_key("Name: Ada Park\nfoo")
    assert ITEM_KEY.fullmatch(key)
    assert item_key("  name:   ADA park foo ") == key
    assert item_key("Name: Ben Ode") != key


def test_split_by_separator_strips_and_drops_empty():
    assert split_by_separator("a one\n---\n\n---\nb two\n---", "---") == ["a one", "b two"]
    assert split_by_separator("only one", "---") == ["only one"]


def test_slice_by_boundaries_is_verbatim_and_ordered():
    packet = "Intro\nName: Ada Park\nfoo\nName: Ben Ode\nbar"
    texts, problems = slice_by_boundaries(packet, ["Name: Ada Park", "Name: Ben Ode"])
    assert texts == ["Name: Ada Park\nfoo\n", "Name: Ben Ode\nbar"] and problems == []


def test_slice_matches_a_normalized_quote_back_to_its_offset():
    packet = "Intro\nName:  Ada   Park, senior\nfoo\nNAME: Ben Ode\nbar"
    texts, problems = slice_by_boundaries(packet, ["name: ada park senior", "Name: Ben Ode"])
    assert problems == []
    assert texts == ["Name:  Ada   Park, senior\nfoo\n", "NAME: Ben Ode\nbar"]


def test_slice_does_not_match_inside_a_word():
    texts, problems = slice_by_boundaries("Canada Parks list\nAda Parks here", ["ada parks"])
    assert problems == [] and texts == ["Ada Parks here"]


def test_slice_reports_missing_ambiguous_and_short_starts():
    packet = "Name: Ada Park\nfoo\nName: Ben Ode\nbar\nName: Ben Ode again"
    _texts, problems = slice_by_boundaries(packet, ["Name: Ada Park", "Name: Cy Lee"])
    assert problems == ["start_quote 2 not found"]
    _texts, problems = slice_by_boundaries(packet, ["Name: Ada Park", "Name: Ben Ode"])
    assert problems == ["start_quote 2 ambiguous"]
    _texts, problems = slice_by_boundaries(packet, ["Name", "Name: Ben Ode again"])
    assert problems == ["start_quote 1 too short"]


def test_slice_searches_forward_only():
    packet = "Name: Ben Ode\nbar\nName: Ada Park\nfoo"
    _texts, problems = slice_by_boundaries(packet, ["Name: Ada Park", "Name: Ben Ode"])
    assert problems == ["start_quote 2 not found"]


def test_slice_with_no_starts_is_not_a_problem():
    assert slice_by_boundaries("anything", []) == ([], [])


def test_build_items_collapses_duplicates():
    items, duplicates = build_items(["A", "B", "A again"], ["Same text", "Other text", " same   TEXT "])
    assert [i["label"] for i in items] == ["A", "B"]
    assert items[0] == {"key": item_key("Same text"), "label": "A", "text": "Same text"}
    assert duplicates == ["A again"]


def _rec(key, state, output=None):
    return {"key": key, "label": key.upper(), "state": state, "output": output}


def test_aggregate_orders_by_roster_and_lists_excluded():
    roster = [{"key": "a", "label": "A", "status": "active", "reason": None},
              {"key": "b", "label": "B", "status": "excluded", "reason": "duplicate"},
              {"key": "c", "label": "C", "status": "active", "reason": None}]
    records = [_rec("c", "completed", {"v": 3}), _rec("a", "completed", {"v": 1})]
    assert aggregate(roster, records) == {
        "items": [{"key": "a", "label": "A", "state": "completed", "output": {"v": 1}},
                  {"key": "c", "label": "C", "state": "completed", "output": {"v": 3}}],
        "excluded": [{"key": "b", "label": "B", "reason": "duplicate"}]}


def test_aggregate_state_precedence():
    assert aggregate_state([_rec("a", "completed"), _rec("b", "completed")]) == ("completed", None)
    assert aggregate_state([_rec("a", "waiting_review"), _rec("b", "failed")]) == ("waiting_approval", None)
    assert aggregate_state([_rec("a", "needs_attention"), _rec("b", "failed"), _rec("c", "completed")]) == (
        "needs_attention", {"code": "items_need_attention", "items": ["a", "b"]})
    assert aggregate_state([_rec("a", "pending"), _rec("b", "completed")]) == ("pending", None)
    assert aggregate_state([]) == ("needs_attention", {"code": "no_items_left"})
