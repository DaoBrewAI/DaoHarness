"""The split_items primitive: one input becomes verbatim, content-keyed items."""
from jsonschema import Draft202012Validator

from daoharness.items import MAX_ITEMS, item_key
from daoharness.primitives import PostContext, builtin_catalog, finish_split

PACKET = "Cover note\nName: Ada Park\nseven years\nName: Ben Ode\nthree years\nName: Cy Lee\nten years"
CONFIG = {"item_noun": "entry"}


def _post(config, output, packet=PACKET):
    p = builtin_catalog().get("split_items")
    return p.post(config, output, PostContext([packet], (), ()))


def _checks(checks):
    return {c.id: c.passed for c in checks}


def _boundaries(*starts):
    return {"items": [{"label": s.split(": ")[-1], "start_quote": s} for s in starts]}


def test_builtin_catalog_lists_split_items():
    p = builtin_catalog().get("split_items")
    assert p is not None and p.executor == "model"
    Draft202012Validator.check_schema(p.config_schema)
    assert not list(Draft202012Validator(p.config_schema).iter_errors({"item_noun": "entry", "max_items": 50,
                                                                        "separator": "---"}))
    assert list(Draft202012Validator(p.config_schema).iter_errors({"max_items": 5}))          # noun required
    assert list(Draft202012Validator(p.config_schema).iter_errors({"item_noun": "e", "max_items": MAX_ITEMS + 1}))


def test_split_post_slices_verbatim_and_keys():
    out, checks = _post(CONFIG, _boundaries("Name: Ada Park", "Name: Ben Ode", "Name: Cy Lee"))
    assert all(c.passed for c in checks)
    assert [i["text"] for i in out["items"]] == ["Name: Ada Park\nseven years\n", "Name: Ben Ode\nthree years\n",
                                                 "Name: Cy Lee\nten years"]
    assert [i["label"] for i in out["items"]] == ["Ada Park", "Ben Ode", "Cy Lee"]
    assert [i["key"] for i in out["items"]] == [item_key(i["text"]) for i in out["items"]]
    assert set(out) == {"items"} and all(set(i) == {"key", "label", "text"} for i in out["items"])


def test_blank_label_falls_back_to_the_noun():
    out, _checks = _post(CONFIG, {"items": [{"label": " ", "start_quote": "Name: Ada Park"}]})
    assert out["items"][0]["label"] == "entry 1"


def test_item_count_fails_on_zero_and_over_max():
    out, checks = _post(CONFIG, {"items": []})
    assert out == {"items": []} and _checks(checks)["item_count"] is False
    out, checks = _post({**CONFIG, "max_items": 2}, _boundaries("Name: Ada Park", "Name: Ben Ode", "Name: Cy Lee"))
    assert _checks(checks)["item_count"] is False
    assert len(out["items"]) == 3  # never truncated silently


def test_duplicate_items_check():
    # Identical items are collapsed to the first; the check passes and records what was collapsed.
    out, duplicates, checks = finish_split(CONFIG, ["A", "B"], ["Name: Ada Park\nfoo\n", "Name: Ada Park\nfoo"], [])
    by_id = {c.id: c for c in checks}
    assert [i["label"] for i in out["items"]] == ["A"] and duplicates == ["B"]
    assert by_id["duplicate_items"].passed and "B" in by_id["duplicate_items"].detail


def test_boundary_failure_check():
    out, checks = _post(CONFIG, _boundaries("Name: Ada Park", "Name: Dee Roe"))
    by_id = {c.id: c for c in checks}
    assert by_id["item_boundaries"].passed is False
    assert "start_quote 2 not found" in by_id["item_boundaries"].detail
    assert out == {"items": []}


def test_separator_mode_builds_items_without_the_model():
    items, _dupes, checks = finish_split({**CONFIG, "separator": "---"}, ["entry 1", "entry 2"], ["a one", "b two"], [])
    assert [i["text"] for i in items["items"]] == ["a one", "b two"] and all(c.passed for c in checks)


def test_more_rows_than_the_hard_cap_fail_item_count():
    # The model schema allows one row past MAX_ITEMS, so an oversized packet fails item_count instead of the last
    # item silently absorbing the rest.
    p = builtin_catalog().get("split_items")
    assert p.output_schema_for({})["properties"]["items"]["maxItems"] == MAX_ITEMS + 1
    names = [f"Entry number {n} start" for n in range(MAX_ITEMS + 1)]
    packet = "\n".join(f"{n}\nbody" for n in names)
    out, checks = _post({"item_noun": "entry", "max_items": MAX_ITEMS},
                        {"items": [{"label": n, "start_quote": n} for n in names]}, packet)
    assert {c.id: c.passed for c in checks}["item_count"] is False and len(out["items"]) == MAX_ITEMS + 1
