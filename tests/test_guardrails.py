from daoharness.guardrails import (
    check_required, check_schema, flatten_text, map_at, mask_sensitive, quote_tokens, substantive, values_at)

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["fields"],
          "properties": {"fields": {"type": "array", "items": {"type": "object", "additionalProperties": False,
              "required": ["name", "source_quote"], "properties": {"name": {"type": "string"}, "source_quote": {"type": "string"}}}}}}


def test_schema_check_reports_path():
    assert check_schema({"fields": []}, SCHEMA).passed
    bad = check_schema({"fields": [{"name": 1, "source_quote": ""}]}, SCHEMA)
    assert not bad.passed and "fields/0/name" in bad.detail


def test_paths():
    data = {"fields": [{"source_quote": "a"}, {"source_quote": "b"}]}
    assert values_at(data, "fields[].source_quote") == ["a", "b"]
    assert map_at(data, "fields[].source_quote", str.upper) == {"fields": [{"source_quote": "A"}, {"source_quote": "B"}]}
    assert data["fields"][0]["source_quote"] == "a"  # not mutated


def test_required_non_empty():
    assert check_required({"subject": "Hi", "body": "x"}, ("subject", "body")).passed
    assert not check_required({"subject": " ", "body": "x"}, ("subject", "body")).passed


def test_mask_sensitive_numbers_but_not_dates():
    out, check = mask_sensitive({"body": "ID 123-45-6789, org 12-3456789, card 4111 1111 1111 1111, "
                                         "acct 000123456789, due 2025-03-14"}, ("body",))
    body = out["body"]
    assert "123-45-6789" not in body and "12-3456789" not in body and "4111 1111 1111 1111" not in body
    assert "000123456789" not in body and "2025-03-14" in body and "6789" in body
    assert check.passed and check.id == "mask_sensitive" and check.detail == "masked 4"


def test_flatten_text_collects_all_strings_in_order():
    assert flatten_text({"a": ["x", {"b": "y"}], "c": 3}) == ["x", "y"]


def test_mask_sensitive_recurses_into_objects_and_lists():
    data = {"meta": {"ref": "ID 123-45-6789", "n": 7, "inner": ["x 987-65-4321"]},
            "rows": [{"note": "123-45-6789"}, {"note": "none"}], "body": "plain"}
    out, check = mask_sensitive(data, ("meta", "rows"))
    assert "123-45-6789" not in str(out) and "987-65-4321" not in str(out)
    assert out["meta"]["n"] == 7 and out["rows"][1]["note"] == "none" and out["body"] == "plain"
    assert check.detail == "masked 3"
    assert data["meta"]["ref"] == "ID 123-45-6789"  # input not mutated


def test_mask_sensitive_list_path_of_objects():
    out, check = mask_sensitive({"rows": [{"a": "123-45-6789"}, {"b": ("1234 5678 9012 3456",)}]}, ("rows[]",))
    assert "123-45-6789" not in str(out) and "1234 5678 9012 3456" not in str(out)
    assert check.detail == "masked 2"


def test_flatten_text_order_on_nested_mixed_structures():
    data = {"a": "1", "b": [{"c": "2", "d": ["3", ("4", {"e": "5"})]}, "6"], "f": {"g": None, "h": "7"}, "i": "8"}
    assert flatten_text(data) == ["1", "2", "3", "4", "5", "6", "7", "8"]


def test_flatten_text_is_linear_on_wide_input():
    import time
    wide = ["x"] * 200_000
    start = time.perf_counter()
    assert len(flatten_text({"a": wide, "b": [wide]})) == 400_000
    assert time.perf_counter() - start < 2.0


def test_quote_tokens_split_cjk_by_character_and_keep_numbers_whole():
    assert quote_tokens("客户提供了身份证") == list("客户提供了身份证")
    assert quote_tokens("Morgan 提交了护照") == ["morgan", "提", "交", "了", "护", "照"]
    assert quote_tokens("3.5 years, $1,500 and 12/03") == ["3.5", "years", "1,500", "and", "12/03"]
    assert quote_tokens("the client\u2019s file") == ["the", "client", "s", "file"]
    assert quote_tokens("...") == []


def test_substantive_quotes_mark_a_place_in_a_text():
    assert substantive("Name: Ada Park") and substantive("身份证和护照")
    assert not substantive("Ada") and not substantive("a b") and not substantive("身份证")
