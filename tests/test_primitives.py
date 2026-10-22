import copy

import pytest
from jsonschema import Draft202012Validator

from daoharness.decision import NONE_OPTION, DecisionOutcome
from daoharness.primitives import (
    BUILTIN_PRIMITIVES, PostContext, PrimitiveCatalog, builtin_catalog, rule_check, rule_items)

IDS = ["collect_inputs", "extract_fields", "checklist_match", "screen_score", "classify_route", "draft_message",
       "generate_questions", "summarize_report", "human_approval", "deliver_package", "generic_model_step",
       "split_items"]
CONFIGS = {
    "collect_inputs": {}, "extract_fields": {"fields": [{"name": "a", "description": "A", "required": True}]},
    "checklist_match": {"items": [{"id": "x", "label": "X", "description": "X"}]},
    "screen_score": {"criteria": [{"id": "c", "label": "C", "description": "C", "weight": 2}], "scale_max": 5},
    "classify_route": {"categories": [{"id": "a", "label": "A", "description": "A"}, {"id": "b", "label": "B", "description": "B"}]},
    "draft_message": {"channel": "email", "audience": "recipient", "external": True},
    "generate_questions": {"max_questions": 3, "focus": "gaps"}, "summarize_report": {"sections": ["Overview"]},
    "human_approval": {}, "deliver_package": {"title": "Package"},
    "generic_model_step": {"output_fields": [{"name": "answer", "description": "Answer"}]},
    "split_items": {"item_noun": "entry"},
}


def _strict(schema):
    if schema.get("type") == "object" and "properties" in schema:
        assert schema["additionalProperties"] is False and sorted(schema["required"]) == sorted(schema["properties"])
        for child in schema["properties"].values():
            _strict(child)
    if schema.get("type") == "array":
        _strict(schema["items"])


def test_exactly_twelve_builtins_in_order():
    assert [p.id for p in BUILTIN_PRIMITIVES] == IDS
    assert builtin_catalog().ids() == IDS


@pytest.mark.parametrize("pid", IDS)
def test_config_and_output_schemas_are_valid_and_strict(pid):
    p = builtin_catalog().get(pid)
    Draft202012Validator.check_schema(p.config_schema)
    assert not list(Draft202012Validator(p.config_schema).iter_errors(CONFIGS[pid]))
    out = p.output_schema_for(CONFIGS[pid])
    Draft202012Validator.check_schema(out)
    if p.executor == "model":
        _strict(out)
        assert p.instructions.strip()


def test_external_draft_flag():
    p = builtin_catalog().get("draft_message")
    assert p.external_draft({"channel": "email", "audience": "x", "external": True})
    assert not p.external_draft({"channel": "document", "audience": "x", "external": False})
    assert not builtin_catalog().get("summarize_report").external_draft({})


def test_checklist_post_checks_coverage_and_keeps_every_status():
    p = builtin_catalog().get("checklist_match")
    cfg = {"items": [{"id": "a", "label": "A", "description": ""}, {"id": "b", "label": "B", "description": ""}]}
    ctx = PostContext(sources=["We attached the photo identification."], input_step_ids=(), completion_criteria=())
    output = {"items": [{"id": "a", "status": "present", "evidence_quote": "photo identification", "note": ""},
                        {"id": "b", "status": "present", "evidence_quote": "", "note": ""}]}
    out, checks = p.post(cfg, output, ctx)
    assert [i["status"] for i in out["items"]] == ["present", "present"] and all(c.passed for c in checks)
    _, checks = p.post(cfg, {"items": [{"id": "a", "status": "missing", "evidence_quote": "", "note": ""}]}, ctx)
    assert any(c.id == "coverage" and not c.passed for c in checks)


def test_only_positive_claims_need_a_quote():
    cat = builtin_catalog()
    checklist = {"items": [{"id": "a", "status": "present", "evidence_quote": "  ", "note": ""},
                           {"id": "b", "status": "missing", "evidence_quote": "", "note": ""},
                           {"id": "c", "status": "unclear", "evidence_quote": "", "note": ""},
                           {"id": "d", "status": "present", "evidence_quote": "Photo ID attached", "note": ""}]}
    assert cat.get("checklist_match").cites(checklist) == ["a"]
    scores = {"scores": [{"criterion_id": "x", "score": 0, "rationale": "", "evidence_quote": ""},
                         {"criterion_id": "y", "score": 2, "rationale": "", "evidence_quote": ""},
                         {"criterion_id": "z", "score": 3, "rationale": "", "evidence_quote": "five years of SQL"}],
              "overall_score": 0, "recommendation": ""}
    assert cat.get("screen_score").cites(scores) == ["y"]
    fields = {"fields": [{"name": "a", "found": True, "value": "v", "source_quotes": []},
                         {"name": "b", "found": False, "value": "", "source_quotes": []},
                         {"name": "c", "found": True, "value": "v", "source_quotes": ["", "a passage"]}]}
    assert cat.get("extract_fields").cites(fields) == ["a"]
    assert cat.get("checklist_match").cites({"items": "not a list"}) == [] and cat.get("screen_score").cites(None) == []


def test_decision_primitives_build_items_and_assemble():
    from daoharness.decision import NONE_OPTION, DecisionOutcome
    cat = builtin_catalog()
    inputs = {"step:assess": {"items": [{"id": "x", "status": "present"}]}, "input:notes": "Photo ID attached."}
    route = cat.get("classify_route")
    [item] = route.decision_items(CONFIGS["classify_route"], inputs, "Pick the branch.")
    assert item.id == "route" and item.options == ("a", "b", NONE_OPTION) and "Photo ID attached." in item.context
    out = route.assemble(CONFIGS["classify_route"], [DecisionOutcome("route", "b", 0.9, "model", False, (), "", ())])
    assert out == {"category_id": "b", "confidence": 0.9, "decided_by": "model"}
    score = cat.get("screen_score")
    [crit] = score.decision_items(CONFIGS["screen_score"], inputs, "")
    assert crit.id == "c" and crit.options == ("0", "1", "2", "3", "4", "5")
    out = score.assemble(CONFIGS["screen_score"], [DecisionOutcome("c", "4", 0.85, "model", False, ("Photo ID",), "seen", ())])
    assert out["scores"][0] == {"criterion_id": "c", "score": 4, "rationale": "seen", "evidence_quote": "Photo ID"}
    check = cat.get("checklist_match")
    [ci] = check.decision_items(CONFIGS["checklist_match"], inputs, "")
    assert ci.options == ("present", "missing", "unclear")
    assert check.executor == route.executor == score.executor == "decision"


def test_screen_score_overall_is_computed_not_trusted():
    p = builtin_catalog().get("screen_score")
    cfg = {"criteria": [{"id": "a", "label": "", "description": "", "weight": 1},
                        {"id": "b", "label": "", "description": "", "weight": 3}], "scale_max": 5}
    out, checks = p.post(cfg, {"scores": [
        {"criterion_id": "a", "score": 5, "rationale": "", "evidence_quote": ""},
        {"criterion_id": "b", "score": 1, "rationale": "", "evidence_quote": ""}],
        "overall_score": 99, "recommendation": ""}, PostContext([], (), ()))
    assert out["overall_score"] == 40.0 and out["recommendation"] == "Overall 40.0/100" and all(c.passed for c in checks)
    _, checks = p.post(cfg, {"scores": [
        {"criterion_id": "a", "score": 9, "rationale": "", "evidence_quote": ""},
        {"criterion_id": "b", "score": 1, "rationale": "", "evidence_quote": ""}],
        "overall_score": 0, "recommendation": ""}, PostContext([], (), ()))
    assert any(c.id == "score_range" and not c.passed for c in checks)


def test_classify_rejects_unknown_category():
    p = builtin_catalog().get("classify_route")
    _, checks = p.post(CONFIGS["classify_route"], {"category_id": "zzz", "confidence": 0.5, "decided_by": "human"},
                       PostContext([], (), ()))
    assert not all(c.passed for c in checks)


def test_summarize_removes_citations_to_steps_that_did_not_feed_it_and_flags_unassessed_criteria():
    p = builtin_catalog().get("summarize_report")
    out = {"title": "t", "summary": "s", "sections": [{"title": "x", "body": "y", "source_steps": ["ghost", "extract"]}],
           "criteria": []}
    fixed, checks = p.post({"sections": []}, out, PostContext([], ("extract",), ("C1",)))
    assert fixed["sections"][0]["source_steps"] == ["extract"]          # never shows invented provenance
    by_id = {c.id: c for c in checks}
    assert by_id["source_steps"].passed and "ghost" in by_id["source_steps"].detail
    assert not by_id["criteria_coverage"].passed


def test_summarize_accepts_prefixed_and_transitively_upstream_citations():
    # A model may cite "step:..." ids, an upstream draft reached through the approval step, or the summary step
    # itself; each is a correct citation.
    p = builtin_catalog().get("summarize_report")
    out = {"title": "t", "summary": "s", "criteria": [{"criterion": "C1", "met": True, "evidence": "e"}],
           "sections": [{"title": "x", "body": "y",
                         "source_steps": ["step:extract", "step:draft", "step:approval", "step:summary", " extract"]}]}
    ctx = PostContext([], ("extract", "approval"), ("C1",), upstream_step_ids=("collect", "extract", "draft", "approval"))
    fixed, checks = p.post({"sections": []}, out, ctx)
    assert fixed["sections"][0]["source_steps"] == ["extract", "draft", "approval"]
    assert all(c.passed for c in checks)
    assert "summary" in {c.id: c for c in checks}["source_steps"].detail


def _valid(schema, value):
    return not list(Draft202012Validator(schema).iter_errors(value))


def _settled(item_id, choice, confidence=0.9, decided_by="model", evidence=(), rationale=""):
    return DecisionOutcome(item_id, choice, confidence, decided_by, False, evidence, rationale, ())


RESUME_TEXT = ("She has built recurring reports, cleaned survey datasets, created dashboards, and presented findings to "
             "program managers. Education: a degree in statistics. Tools: spreadsheet software, SQL-like query tools.")


def test_extract_quotes_are_a_list_of_passages():
    p = builtin_catalog().get("extract_fields")
    row = p.output_schema_for({})["properties"]["fields"]["items"]
    assert row["properties"]["source_quotes"] == {"type": "array", "items": {"type": "string", "maxLength": 2000},
                                                 "minItems": 0, "maxItems": 8}
    assert "source_quote" not in row["properties"]
    assert "own passage" in p.instructions and "never join" in p.instructions
    two_places = {"fields": [{"name": "skills", "found": True, "value": "reports, dashboards, SQL",
                              "source_quotes": ["presented findings to program managers",
                                                "Tools: spreadsheet software, SQL-like query tools"]}]}
    assert p.cites(two_places) == []
    assert all(q in RESUME_TEXT for q in two_places["fields"][0]["source_quotes"])


def test_extract_merges_a_field_returned_once_per_passage():
    # A model may return the same field once per passage: the rows merge into one.
    p = builtin_catalog().get("extract_fields")
    config = {"fields": [{"name": "skills", "description": "S", "required": True},
                         {"name": "name", "description": "N", "required": True}]}
    output = {"fields": [{"name": "skills", "found": True, "value": "SQL", "source_quotes": ["SQL reporting"]},
                         {"name": "name", "found": True, "value": "Ada", "source_quotes": ["Ada"]},
                         {"name": "skills", "found": True, "value": "dashboards", "source_quotes": ["dashboards"]},
                         {"name": "skills", "found": False, "value": "", "source_quotes": []},
                         {"name": "skills", "found": True, "value": "SQL", "source_quotes": ["SQL reporting"]}]}
    out, checks = p.post(config, output, PostContext(["SQL reporting, dashboards. Ada"], (), ()))
    assert out["fields"] == [
        {"name": "skills", "found": True, "value": "SQL; dashboards", "source_quotes": ["SQL reporting", "dashboards"]},
        {"name": "name", "found": True, "value": "Ada", "source_quotes": ["Ada"]}]
    assert all(c.passed for c in checks)


def test_extract_keeps_flagging_an_item_that_holds_several_records():
    # Three records in one item make every field come back three times: that still fails coverage instead of
    # merging into "Jordan; Priya; Mateo".
    p = builtin_catalog().get("extract_fields")
    config = {"fields": [{"name": "name", "description": "N", "required": True},
                         {"name": "title", "description": "T", "required": True}]}
    rows = [{"name": n, "found": True, "value": v, "source_quotes": [v]}
            for v1, v2 in (("Jordan", "Designer"), ("Priya", "Engineer")) for n, v in (("name", v1), ("title", v2))]
    out, checks = p.post(config, {"fields": rows}, PostContext(["Jordan Designer Priya Engineer"], (), ()))
    assert len(out["fields"]) == 4
    coverage = next(c for c in checks if c.id == "coverage")
    assert not coverage.passed and "duplicated" in coverage.detail


def test_extract_found_field_needs_a_value_and_a_passage():
    p = builtin_catalog().get("extract_fields")
    config = {"fields": [{"name": "a", "description": "A", "required": True},
                         {"name": "b", "description": "B", "required": True}]}
    output = {"fields": [{"name": "a", "found": True, "value": "x", "source_quotes": ["x y z"]},
                         {"name": "b", "found": True, "value": "y", "source_quotes": ["  "]}]}
    _out, checks = p.post(config, output, PostContext(["x y z"], (), ()))
    assert next(c for c in checks if c.id == "found_fields").detail == "b"


def test_describe_has_contract_keys_and_catalog_describe_lists_all():
    d = builtin_catalog().get("extract_fields").describe()
    assert set(d) == {"id", "version", "title", "purpose", "executor", "config_schema"}
    assert d["version"] == 1
    assert [x["id"] for x in builtin_catalog().describe()] == IDS


def test_catalog_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        PrimitiveCatalog([BUILTIN_PRIMITIVES[0], BUILTIN_PRIMITIVES[0]])
    assert builtin_catalog().get("nope") is None


def test_executors_and_guardrail_defaults():
    cat = builtin_catalog()
    executors = {pid: cat.get(pid).executor for pid in IDS}
    assert executors == {"collect_inputs": "deterministic", "extract_fields": "model", "checklist_match": "decision",
                         "screen_score": "decision", "classify_route": "decision", "draft_message": "model",
                         "generate_questions": "model", "summarize_report": "model", "human_approval": "human",
                         "deliver_package": "deterministic", "generic_model_step": "model",
                         "split_items": "model"}
    assert cat.get("extract_fields").default_guardrails == ("schema", "citations")
    assert {pid for pid in IDS if cat.get(pid).cites is not None} == {
        "extract_fields", "checklist_match", "screen_score"}
    draft = cat.get("draft_message")
    assert draft.default_guardrails == ("schema", "required_fields", "mask_sensitive")
    assert draft.required_paths == ("subject", "body") and draft.text_paths == ("subject", "body")
    assert cat.get("human_approval").default_guardrails == ()
    for pid in IDS:
        p = cat.get(pid)
        assert bool(p.instructions.strip()) == (p.executor == "model")
        assert bool(p.rubric.strip()) == (p.executor == "decision")
        if p.executor == "decision":
            assert p.decision_items is not None and p.assemble is not None


def test_required_paths_for_generic_model_step_follow_config():
    cat = builtin_catalog()
    cfg = {"output_fields": [{"name": "answer", "description": ""}, {"name": "reason", "description": ""}]}
    assert cat.get("generic_model_step").required_paths_for(cfg) == ("answer", "reason")
    assert cat.get("summarize_report").required_paths_for({"sections": []}) == ("title", "summary")


def test_config_schemas_reject_bad_ids_and_unknown_keys():
    cat = builtin_catalog()
    checklist = cat.get("checklist_match").config_schema
    assert not _valid(checklist, {"items": [{"id": "x\n", "label": "X", "description": "X"}]})
    assert not _valid(checklist, {"items": [{"id": "Bad", "label": "X", "description": "X"}]})
    assert not _valid(checklist, {"items": [{"id": "x", "label": "X", "description": "X"}], "extra": 1})
    assert not _valid(cat.get("extract_fields").config_schema, {"fields": []})
    assert not _valid(cat.get("classify_route").config_schema,
                      {"categories": [{"id": "a", "label": "A", "description": "A"}]})


@pytest.mark.parametrize("pid", ["checklist_match", "screen_score", "classify_route"])
def test_decision_threshold_is_optional_and_bounded(pid):
    schema = builtin_catalog().get(pid).config_schema
    assert _valid(schema, CONFIGS[pid])
    assert _valid(schema, {**CONFIGS[pid], "decision_threshold": 0.9})
    assert not _valid(schema, {**CONFIGS[pid], "decision_threshold": 0.3})
    assert not _valid(schema, {**CONFIGS[pid], "decision_threshold": 1.0})


def test_draft_message_rules_are_optional_and_bounded():
    schema = builtin_catalog().get("draft_message").config_schema
    assert _valid(schema, {**CONFIGS["draft_message"], "rules": ["Mention the deadline."]})
    assert not _valid(schema, {**CONFIGS["draft_message"], "rules": ["r"] * 11})
    assert not _valid(schema, {"channel": "email", "audience": "x"})


def test_decision_items_context_is_flattened_inputs_truncated():
    cat = builtin_catalog()
    inputs = {"input:notes": "A" * 30000}
    [item] = cat.get("checklist_match").decision_items(CONFIGS["checklist_match"], inputs, "")
    assert len(item.context) == 24000
    for pid in ("screen_score", "classify_route"):
        [cut] = cat.get(pid).decision_items(CONFIGS[pid], inputs, "")
        [whole] = cat.get(pid).decision_items(CONFIGS[pid], {"a": "x"}, "")
        assert len(cut.context) == 24000 and whole.context == "x"
    [item] = cat.get("checklist_match").decision_items(CONFIGS["checklist_match"], {"a": "one", "b": ["two"]}, "")
    assert item.context == "one\n\n[[next input]]\n\ntwo" and item.question == "Do the inputs show this item? X: X"
    [route] = cat.get("classify_route").decision_items(CONFIGS["classify_route"], {"a": "x"}, "Pick the branch.")
    assert route.question.startswith("Which category fits these inputs best?") and "Pick the branch." in route.question
    assert route.option_descriptions == {"a": "A: A", "b": "B: B", NONE_OPTION: "None of the categories fits"}
    [crit] = cat.get("screen_score").decision_items(CONFIGS["screen_score"], {"a": "x"}, "")
    assert crit.question == "Score this criterion from 0 to 5. C: C"
    assert crit.option_descriptions == {"0": "no evidence", "5": "fully meets"}


def test_checklist_assemble_notes_and_output_validates():
    p = builtin_catalog().get("checklist_match")
    cfg = {"items": [{"id": "a", "label": "A", "description": ""}, {"id": "b", "label": "B", "description": ""}]}
    out = p.assemble(cfg, [_settled("a", "present", 0.93, "model"),
                           _settled("b", "missing", None, "human")])
    assert out == {"items": [{"id": "a", "status": "present", "evidence_quote": "", "note": "model 0.93"},
                             {"id": "b", "status": "missing", "evidence_quote": "", "note": "human"}]}
    assert _valid(p.output_schema_for(cfg), out)
    out = p.assemble(cfg, [_settled("a", "present", 0.88, "model", ("photo id",)), _settled("b", "unclear")])
    assert out["items"][0] == {"id": "a", "status": "present", "evidence_quote": "photo id", "note": "model 0.88"}


def test_classify_assemble_and_post_for_human_none_apply():
    p = builtin_catalog().get("classify_route")
    out = p.assemble(CONFIGS["classify_route"], [_settled("route", NONE_OPTION, None, "human")])
    assert out == {"category_id": NONE_OPTION, "confidence": 0.0, "decided_by": "human"}
    assert _valid(p.output_schema_for(CONFIGS["classify_route"]), out)
    _, checks = p.post(CONFIGS["classify_route"], out, PostContext([], (), ()))
    assert [(c.id, c.passed) for c in checks] == [("category", False)]
    _, checks = p.post(CONFIGS["classify_route"], {"category_id": "a", "confidence": 0.9, "decided_by": "model"},
                       PostContext([], (), ()))
    assert [(c.id, c.passed) for c in checks] == [("category", True)]
    _, checks = p.post(CONFIGS["classify_route"], {"category_id": "a", "confidence": 1.5, "decided_by": "model"},
                       PostContext([], (), ()))
    assert not checks[0].passed


def test_classify_assemble_without_route_outcome_fails_category_check_instead_of_raising():
    p = builtin_catalog().get("classify_route")
    out = p.assemble(CONFIGS["classify_route"], [])
    assert out == {"category_id": "", "confidence": 0.0, "decided_by": ""}
    assert _valid(p.output_schema_for(CONFIGS["classify_route"]), out)
    _, checks = p.post(CONFIGS["classify_route"], out, PostContext([], (), ()))
    assert [(c.id, c.passed) for c in checks] == [("category", False)]


def test_screen_score_assembled_output_validates_after_post():
    p = builtin_catalog().get("screen_score")
    out = p.assemble(CONFIGS["screen_score"], [_settled("c", "3", 0.9, "model")])
    out, checks = p.post(CONFIGS["screen_score"], out, PostContext([], (), ()))
    assert out["overall_score"] == 60.0 and out["recommendation"] == "Overall 60.0/100"
    assert _valid(p.output_schema_for(CONFIGS["screen_score"]), out) and all(c.passed for c in checks)


def test_post_does_not_mutate_its_input():
    p = builtin_catalog().get("extract_fields")
    cfg = {"fields": [{"name": "a", "description": "A", "required": True}]}
    raw = {"fields": [{"name": "a", "found": True, "value": "x", "source_quotes": ["x one"]},
                      {"name": "a", "found": True, "value": "y", "source_quotes": ["y two"]}]}
    before = copy.deepcopy(raw)
    out, _ = p.post(cfg, raw, PostContext(["source"], (), ()))
    assert len(out["fields"]) == 1 and out["fields"][0]["value"] == "x; y" and raw == before


def test_extract_fields_post_checks_coverage_and_found_fields():
    p = builtin_catalog().get("extract_fields")
    cfg = {"fields": [{"name": "a", "description": "", "required": True},
                      {"name": "b", "description": "", "required": False}]}
    ctx = PostContext(["x"], (), ())
    good = {"fields": [{"name": "a", "found": True, "value": "v", "source_quotes": ["x"]},
                       {"name": "b", "found": False, "value": "", "source_quotes": []}]}
    _, checks = p.post(cfg, good, ctx)
    assert {c.id: c.passed for c in checks} == {"coverage": True, "found_fields": True}
    dup = {"fields": [good["fields"][0], good["fields"][0]]}
    _, checks = p.post(cfg, dup, ctx)
    assert {c.id: c.passed for c in checks}["coverage"] is False
    hollow = {"fields": [{"name": "a", "found": True, "value": " ", "source_quotes": ["x"]}, good["fields"][1]]}
    _, checks = p.post(cfg, hollow, ctx)
    assert {c.id: c.passed for c in checks} == {"coverage": True, "found_fields": False}


def test_generate_questions_limit():
    p = builtin_catalog().get("generate_questions")
    q = {"question": "q", "reason": "r"}
    _, checks = p.post({"max_questions": 2, "focus": ""}, {"questions": [q, q]}, PostContext([], (), ()))
    assert [(c.id, c.passed) for c in checks] == [("question_limit", True)]
    _, checks = p.post({"max_questions": 2, "focus": ""}, {"questions": [q, q, q]}, PostContext([], (), ()))
    assert [(c.id, c.passed) for c in checks] == [("question_limit", False)]


def test_summarize_passes_with_known_steps_and_assessed_criteria():
    p = builtin_catalog().get("summarize_report")
    out = {"title": "t", "summary": "s", "sections": [{"title": "x", "body": "y", "source_steps": ["extract"]}],
           "criteria": [{"criterion": "c1", "met": True, "evidence": "e"}]}
    _, checks = p.post({"sections": []}, out, PostContext([], ("extract",), ("C1",)))
    assert all(c.passed for c in checks) and {c.id for c in checks} == {"source_steps", "criteria_coverage"}


def test_draft_rule_items_and_compliance_check():
    cfg = {**CONFIGS["draft_message"], "rules": ["Mention the reply date.", "Stay under 200 words."]}
    output = {"subject": "Next steps", "body": "Please reply by Friday.", "notes": ""}
    items = rule_items(cfg, output)
    assert [i.id for i in items] == ["rule_1", "rule_2"]
    assert items[0].kind == "yes_no" and items[0].options == ("yes", "no")
    assert items[0].question == "Does this draft comply with the rule: Mention the reply date.?"
    assert "Next steps" in items[0].context and "Please reply by Friday." in items[0].context
    assert rule_items(CONFIGS["draft_message"], output) == []
    ok = rule_check(cfg, [_settled("rule_1", "yes"), _settled("rule_2", "yes")])
    assert ok.id == "rule_compliance" and ok.passed
    bad = rule_check(cfg, [_settled("rule_1", "no"),
                           DecisionOutcome("rule_2", None, None, None, True, (), "", ())])
    assert not bad.passed and "Mention the reply date." in bad.detail and "Stay under 200 words." in bad.detail
    missing = rule_check(cfg, [_settled("rule_1", "yes")])
    assert not missing.passed and "Stay under 200 words." in missing.detail


def _categories(n):
    return {"categories": [{"id": f"c{i}", "label": f"C{i}", "description": f"C{i}"} for i in range(n)]}


def test_classify_route_caps_categories_so_options_fit_a_decision_item():
    p = builtin_catalog().get("classify_route")
    assert not _valid(p.config_schema, _categories(16))
    assert _valid(p.config_schema, _categories(15))
    [item] = p.decision_items(_categories(15), {"a": "x"}, "Pick the branch.")
    assert len(item.options) == 16 and item.options[-1] == NONE_OPTION
