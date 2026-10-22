"""What the engine says when a model call or a build goes wrong: a sentence that names the problem and the next step.

The codes stay in the records (a ModelError's code, a validation error's "code:step_id:detail"); these sentences are
what a person reads.
"""
from __future__ import annotations

import re
from typing import Iterable

# A ModelError's code -> (what went wrong, what to do).
MODEL_ERRORS: dict[str, tuple[str, str]] = {
    "provider_rejected": (
        "The model provider rejected the request.",
        "Check that DAOHARNESS_MODEL is a model it serves, that your key is right and that DAOHARNESS_BASE_URL is "
        "its address, then try again."),
    "provider_auth": (
        "The model provider did not accept your key.",
        "Check DAOHARNESS_API_KEY (or the key variable for your provider) and that the key may use this model."),
    "model_not_found": (
        "The model provider does not know the model in DAOHARNESS_MODEL.",
        "Check its name, and that DAOHARNESS_BASE_URL is the address of the provider that serves it."),
    "out_of_credit": (
        "The model provider says you are out of credit.",
        "Add credit with your provider or use another key, then try again."),
    "provider_rate_limited": ("The model provider is limiting requests.", "Wait a minute and try again."),
    "provider_failed": ("The model provider had an error on its side.", "Try again in a minute."),
    "provider_unreachable": (
        "Could not reach the model provider.",
        "Check your internet connection and DAOHARNESS_BASE_URL, then try again."),
    "provider_tls": (
        "The secure connection to the model provider failed.",
        "Check that DAOHARNESS_BASE_URL is right (https or http) and that this computer trusts the provider's "
        "certificate."),
    "provider_timeout": (
        "The model provider did not answer in time.",
        "A slow or busy model can take longer than we wait: try again, set DAOHARNESS_TIMEOUT to the seconds to wait, "
        "or use a faster model."),
    "provider_bad_reply": (
        "The model provider's answer was not in the format an OpenAI-compatible endpoint sends.",
        "Check that DAOHARNESS_BASE_URL is its API address (it usually ends in /v1)."),
    "invalid_model_json": (
        "The model's reply was not valid JSON.",
        "Try again; a larger model usually follows the format more reliably."),
    "reply_truncated": (
        "The model's reply was cut off before it finished.",
        "Raise DAOHARNESS_MAX_TOKENS (or leave it unset) and try again."),
    "invalid_decision_output": (
        "The model's answers to a set of choices could not be used.",
        "Try again; a larger model usually follows the format more reliably."),
}
SHOWN_PROBLEMS = 3


def model_error_message(code: str, detail: str = "") -> str:
    """What went wrong with a model call, with `detail` in brackets, and what to do about it."""
    known = MODEL_ERRORS.get(code)
    if known is None:
        shown = f"{code}; {detail}" if detail else code
        return f"The model could not be used ({shown}). Check your model settings and try again."
    what, todo = known
    return f"{what[:-1]} ({detail}). {todo}" if detail else f"{what} {todo}"


def validation_message(errors: Iterable[str]) -> str:
    """A workflow the model wrote twice and the checks still refused: the first few problems, each with its step."""
    problems = validation_problems(errors)
    lead = "The workflow the model wrote did not pass validation, even on a second try"
    return f"{lead}: {problems}. Try again, or describe the job in more detail." if problems else (
        f"{lead}. Try again, or describe the job in more detail.")


def validation_problems(errors: Iterable[str], limit: int = SHOWN_PROBLEMS) -> str:
    """The first `limit` validation errors as sentences, then how many more there are."""
    errors = list(errors)
    explained = {e.split(":", 2)[1] for e in errors if e.startswith("config_json_problem:")}
    sentences = [describe_validation_error(e) for e in errors
                 if not (e.startswith("invalid_config_json:") and e.split(":", 1)[1] in explained)]
    more = f" (+{len(sentences) - limit} more)" if len(sentences) > limit else ""
    return "; ".join(sentences[:limit]) + more


# What a validation error says about its step, by code. {d} is the detail after the step id.
STEP_PROBLEMS = {
    "unknown_input": "reads an input called '{d}' that the workflow does not list",
    "forward_or_unknown_step": "reads step '{d}', which does not come before it",
    "unknown_primitive": "uses '{d}', which is not a step type this engine has",
    "invalid_config": "has settings its step type does not accept ({d})",
    "approver_not_member": "names an approver, '{d}', who cannot be found",
    "approval_without_content": "asks for a sign-off but reads nothing to approve",
    "external_draft_requires_approval": "drafts a message to someone outside, and no later sign-off step reads it",
    "missing_inputs": "({d}) reads nothing; it needs at least one input or earlier step",
}
# Codes whose detail is already a sentence about the step.
SENTENCE_DETAILS = frozenset({
    "invalid_for_each", "not_per_item", "multiple_splits", "split_input", "split_without_items",
    "segment_not_contiguous", "segment_too_long", "per_item_input"})
APPROVALS = {"approval required": "is a sign-off step but names no approver",
             "approval not allowed": "names an approver but is not a sign-off step"}
SCHEMA_STEP = re.compile(r"spec/steps/(\d+)(?:/(.*))?")
SPEC_STEP_ID = re.compile(r"steps\[(\d+)\]\.id")


def describe_validation_error(error: str) -> str:
    """One validation error ("code:step_id:detail") as a sentence that names the step. An error this does not know
    is returned as it is."""
    code, _, rest = error.partition(":")
    if code == "invalid_output":   # the reply does not fit the schema: invalid_output:<path>:<message>
        where, _, message = rest.partition(":")
        step = SCHEMA_STEP.fullmatch(where)
        if step:
            field = step.group(2)
            return f"step {int(step.group(1)) + 1}{f', {field}' if field else ''}: {message}"
        return f"the reply does not fit the expected shape at {where}: {message}"
    if code == "invalid_spec":
        return _spec_problem(rest)
    step_id, _, detail = rest.partition(":")
    detail = detail.strip()
    if code == "config_json_problem":
        return f"step '{step_id}' has settings (config_json) that are not usable: {detail}"
    if code == "invalid_config_json":
        return f"step '{step_id}' has settings (config_json) that are not a JSON object"
    if code == "missing_deliverable":
        return f"the last step, '{step_id}', must be a deliver_package step"
    if code == "approval_mismatch" and detail in APPROVALS:
        return f"step '{step_id}' {APPROVALS[detail]}"
    if code in SENTENCE_DETAILS:
        return f"step '{step_id}': {detail}"
    if code in STEP_PROBLEMS and step_id:
        problem = STEP_PROBLEMS[code].format(d=detail)
        return f"step '{step_id}' {problem}"
    return error


def _spec_problem(path: str) -> str:
    kind, _, name = path.partition(":")
    if kind == "duplicate_step_id":
        return f"two steps share the id '{name}'"
    if kind == "duplicate_input":
        return f"two inputs share the name '{name}'"
    step = SPEC_STEP_ID.fullmatch(path)
    if step:
        return (f"step {int(step.group(1)) + 1} has no id, or one that is not lower-case letters, digits and "
                "underscores")
    return f"the workflow has a missing or invalid value at {path}"
