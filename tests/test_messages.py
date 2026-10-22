"""What the engine says when something goes wrong: a sentence that names the problem and the next step."""
import pytest

from daoharness.messages import model_error_message, validation_message, validation_problems

MODEL_SENTENCES = {
    "provider_rejected": (
        "The model provider rejected the request. Check that DAOHARNESS_MODEL is a model it serves, that your key "
        "is right and that DAOHARNESS_BASE_URL is its address, then try again."),
    "provider_auth": (
        "The model provider did not accept your key. Check DAOHARNESS_API_KEY (or the key variable for your "
        "provider) and that the key may use this model."),
    "model_not_found": (
        "The model provider does not know the model in DAOHARNESS_MODEL. Check its name, and that "
        "DAOHARNESS_BASE_URL is the address of the provider that serves it."),
    "out_of_credit": (
        "The model provider says you are out of credit. Add credit with your provider or use another key, then "
        "try again."),
    "provider_rate_limited": "The model provider is limiting requests. Wait a minute and try again.",
    "provider_failed": "The model provider had an error on its side. Try again in a minute.",
    "provider_unreachable": (
        "Could not reach the model provider. Check your internet connection and DAOHARNESS_BASE_URL, then try "
        "again."),
    "provider_tls": (
        "The secure connection to the model provider failed. Check that DAOHARNESS_BASE_URL is right (https or "
        "http) and that this computer trusts the provider's certificate."),
    "provider_timeout": (
        "The model provider did not answer in time. A slow or busy model can take longer than we wait: try again, "
        "set DAOHARNESS_TIMEOUT to the seconds to wait, or use a faster model."),
    "provider_bad_reply": (
        "The model provider's answer was not in the format an OpenAI-compatible endpoint sends. Check that "
        "DAOHARNESS_BASE_URL is its API address (it usually ends in /v1)."),
    "invalid_model_json": (
        "The model's reply was not valid JSON. Try again; a larger model usually follows the format more "
        "reliably."),
    "reply_truncated": (
        "The model's reply was cut off before it finished. Raise DAOHARNESS_MAX_TOKENS (or leave it unset) and try "
        "again."),
    "invalid_decision_output": (
        "The model's answers to a set of choices could not be used. Try again; a larger model usually follows the "
        "format more reliably."),
}


@pytest.mark.parametrize("code", MODEL_SENTENCES)
def test_every_model_error_has_a_sentence_that_says_what_to_do(code):
    assert model_error_message(code) == MODEL_SENTENCES[code]


def test_a_detail_goes_inside_the_first_sentence():
    assert model_error_message("invalid_model_json", "extra data after the JSON object") == (
        "The model's reply was not valid JSON (extra data after the JSON object). Try again; a larger model "
        "usually follows the format more reliably.")
    assert model_error_message("provider_rejected", "HTTP 400").startswith(
        "The model provider rejected the request (HTTP 400). Check that")


def test_a_code_the_engine_does_not_know_still_gets_a_sentence():
    assert model_error_message("weird_code") == (
        "The model could not be used (weird_code). Check your model settings and try again.")
    assert model_error_message("weird_code", "oops") == (
        "The model could not be used (weird_code; oops). Check your model settings and try again.")


def test_a_sentence_names_the_setting_to_check():
    for code in ("provider_rejected", "provider_auth", "model_not_found", "provider_unreachable", "provider_tls"):
        assert "DAOHARNESS_" in MODEL_SENTENCES[code]
    assert "DAOHARNESS_MAX_TOKENS" in MODEL_SENTENCES["reply_truncated"]


# --- the workflow the model wrote -----------------------------------------------------------------------------------

SENTENCES = [
    ("unknown_input:collect:docs", "step 'collect' reads an input called 'docs' that the workflow does not list"),
    ("forward_or_unknown_step:assess:nowhere", "step 'assess' reads step 'nowhere', which does not come before it"),
    ("unknown_primitive:assess:magic", "step 'assess' uses 'magic', which is not a step type this engine has"),
    ("invalid_config:assess:items must not be empty",
     "step 'assess' has settings its step type does not accept (items must not be empty)"),
    ("approval_mismatch:sign:approval required", "step 'sign' is a sign-off step but names no approver"),
    ("approval_mismatch:draft:approval not allowed", "step 'draft' names an approver but is not a sign-off step"),
    ("approver_not_member:sign:Partner", "step 'sign' names an approver, 'Partner', who cannot be found"),
    ("approval_without_content:sign:no inputs", "step 'sign' asks for a sign-off but reads nothing to approve"),
    ("external_draft_requires_approval:draft:draft_message",
     "step 'draft' drafts a message to someone outside, and no later sign-off step reads it"),
    ("missing_inputs:collect:collect_inputs", "step 'collect' (collect_inputs) reads nothing; it needs at least one "
                                              "input or earlier step"),
    ("missing_deliverable:summary:last step must be deliver_package",
     "the last step, 'summary', must be a deliver_package step"),
    ("invalid_for_each:extract:split is not an earlier split_items step",
     "step 'extract': split is not an earlier split_items step"),
    ("segment_not_contiguous:route:per-item steps must directly follow split, with no other step between them",
     "step 'route': per-item steps must directly follow split, with no other step between them"),
    ("invalid_config_json:extract", "step 'extract' has settings (config_json) that are not a JSON object"),
    ("config_json_problem:extract: not valid JSON (syntax error: Expecting ',' delimiter (line 1, column 9))",
     "step 'extract' has settings (config_json) that are not usable: not valid JSON (syntax error: Expecting ',' "
     "delimiter (line 1, column 9))"),
    ("invalid_spec:duplicate_step_id:extract", "two steps share the id 'extract'"),
    ("invalid_spec:duplicate_input:docs", "two inputs share the name 'docs'"),
    ("invalid_spec:steps[2].id", "step 3 has no id, or one that is not lower-case letters, digits and underscores"),
    ("invalid_spec:input_requirements[0].description",
     "the workflow has a missing or invalid value at input_requirements[0].description"),
    ("invalid_output:spec/steps/2/inputs/0:'step:Nowhere' does not match '^(input|step):[a-z]'",
     "step 3, inputs/0: 'step:Nowhere' does not match '^(input|step):[a-z]'"),
    ("invalid_output:spec/domain:None is not of type 'string'",
     "the reply does not fit the expected shape at spec/domain: None is not of type 'string'"),
    ("something_new:step_x:detail here", "something_new:step_x:detail here"),
]


@pytest.mark.parametrize("error, sentence", SENTENCES)
def test_each_validation_error_becomes_a_sentence_with_its_step(error, sentence):
    assert validation_message([error]) == (
        f"The workflow the model wrote did not pass validation, even on a second try: {sentence}. "
        "Try again, or describe the job in more detail.")


def test_only_the_first_few_problems_are_shown_and_the_rest_counted():
    errors = [f"unknown_input:s{n}:x" for n in range(1, 8)]
    message = validation_message(errors)
    assert "step 's1'" in message and "step 's3'" in message and "step 's4'" not in message
    assert "(+4 more)" in message
    assert validation_message(errors[:3]).count("(+") == 0
    assert "step 's5'" in validation_problems(errors, limit=5) and "(+2 more)" in validation_problems(errors, limit=5)


def test_a_config_json_hint_replaces_the_bare_error_for_the_same_step():
    errors = ["config_json_problem:extract: not valid JSON (no JSON object found in the reply)",
              "invalid_config_json:extract", "invalid_config_json:assess"]
    message = validation_message(errors)
    assert message.count("step 'extract'") == 1 and "step 'assess' has settings (config_json)" in message
