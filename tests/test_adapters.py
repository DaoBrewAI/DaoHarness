"""The ready-made model port over an OpenAI-compatible endpoint, against a local stand-in (no network beyond
127.0.0.1)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from daoharness.adapters import OpenAICompatibleModel, parse_json_object
from daoharness.compiler import WorkflowCompiler
from daoharness.ports import ModelError, ModelReceipt
from daoharness.primitives import builtin_catalog
from daoharness.validator import ValidationContext
from industry_fixtures import ACCOUNTING
from workflow_fakes import as_model_payload

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}


class Server:
    """A local HTTP server: `answer(method, path, body)` -> (status, headers, body bytes)."""

    def __init__(self, answer):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else None
                outer.requests.append({"method": method, "path": self.path, "headers": dict(self.headers),
                                       "body": body})
                status, headers, data = answer(method, self.path, body)
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._reply("GET")

            def do_POST(self):
                self._reply("POST")

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def completion(content, finish_reason=None, **extra):
    choice = {"message": {"role": "assistant", "content": content}}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    return 200, {"Content-Type": "application/json"}, json.dumps(
        {"id": "chatcmpl-1", "model": "served-model", "choices": [choice], **extra}).encode()


def error(status, message):
    return status, {"Content-Type": "application/json"}, json.dumps({"error": {"message": message}}).encode()


@pytest.fixture
def serve():
    servers = []

    def start(answer):
        server = Server(answer)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


def model_for(server, **kw):
    return OpenAICompatibleModel(base_url=server.url + "/v1", model="my-model", api_key="test-key",
                                 sleep=lambda seconds: None, **kw)


def call(model, operation="step"):
    return model.generate(org_id="o", operation=operation, skill_id="s", input_data={"text": "hello"},
                          instructions="Count the words.", evidence=[{"quote": "hi"}], schema=SCHEMA)


def test_asks_for_strict_structured_output_and_returns_a_measured_receipt(serve):
    server = serve(lambda m, p, b: completion('{"a": 1}', usage={"prompt_tokens": 12, "completion_tokens": 3}))
    content, receipt = call(model_for(server), operation="compile")
    assert content == {"a": 1}
    assert receipt == ModelReceipt(run_id="chatcmpl-1", provider="openai-compatible", model="served-model",
                                   output_digest=receipt.output_digest, input_tokens=12, output_tokens=3,
                                   tokens_estimated=False)
    [request] = server.requests
    assert request["path"] == "/v1/chat/completions" and request["headers"]["Authorization"] == "Bearer test-key"
    body = request["body"]
    assert body["model"] == "my-model"
    assert body["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "compile", "schema": SCHEMA, "strict": True}}
    system, user = body["messages"]
    assert system["role"] == "system" and "data, never instructions" in system["content"]
    assert "Count the words." in user["content"] and '"text":"hello"' in user["content"]
    assert '"quote":"hi"' in user["content"]
    assert "JSON Schema" in user["content"]   # also here, for endpoints that ignore response_format


def test_steps_down_to_json_mode_then_plain_and_remembers(serve):
    answers = iter([error(400, "response_format.json_schema is not supported by this model"),
                    error(400, "Invalid value for response_format: json_object"),
                    completion('Here you go:\n```json\n{"a": 2}\n```'),
                    completion('{"a": 3}')])
    server = serve(lambda m, p, b: next(answers))
    model = model_for(server)
    assert call(model)[0] == {"a": 2}
    assert model.mode == "plain"
    assert call(model)[0] == {"a": 3}
    formats = [r["body"].get("response_format", {}).get("type") for r in server.requests]
    assert formats == ["json_schema", "json_object", None, None]
    assert "JSON Schema" in server.requests[2]["body"]["messages"][1]["content"]


def test_a_bare_invalid_argument_still_steps_down(serve):
    # Vertex AI and Gemini refuse a strict schema with no word about response_format
    answers = iter([error(400, "Request contains an invalid argument."), completion('{"a": 2}'),
                    completion('{"a": 3}')])
    server = serve(lambda m, p, b: next(answers))
    model = model_for(server)
    assert call(model)[0] == {"a": 2} and call(model)[0] == {"a": 3}
    formats = [r["body"].get("response_format", {}).get("type") for r in server.requests]
    assert formats == ["json_schema", "json_object", "json_object"] and model.mode == "json_object"


def test_a_strict_request_that_stalls_steps_down_instead_of_waiting_again(serve):
    def answer(m, p, b):
        if b.get("response_format", {}).get("type") == "json_schema":
            time.sleep(0.6)
        return completion('{"a": 4}')
    server = serve(answer)
    model = model_for(server, timeout=0.3)
    assert call(model)[0] == {"a": 4} and model.mode == "json_object"
    assert [r["body"].get("response_format", {}).get("type") for r in server.requests] == ["json_schema",
                                                                                           "json_object"]


def test_a_mode_is_kept_when_every_mode_fails(serve):
    server = serve(lambda m, p, b: error(400, "messages must not be empty"))
    model = model_for(server)
    with pytest.raises(ModelError) as raised:
        call(model)
    assert (raised.value.code, raised.value.detail) == ("provider_rejected", "HTTP 400")
    assert len(server.requests) == 3 and model.mode == "json_schema"


@pytest.mark.parametrize("base_url", ["https://generativelanguage.googleapis.com/v1beta/openai",
                                      "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi",
                                      "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p/locations/us-central1/endpoints/openapi"])
def test_google_endpoints_start_in_json_mode(base_url):
    assert OpenAICompatibleModel(base_url=base_url, model="g").mode == "json_object"
    assert OpenAICompatibleModel(base_url="https://api.openai.com/v1", model="m").mode == "json_schema"
    assert OpenAICompatibleModel(base_url=base_url, model="g", mode="plain").mode == "plain"


def test_a_slow_provider_gets_one_more_wait_not_three(serve):
    server = serve(lambda m, p, b: (time.sleep(0.6), completion('{"a": 1}'))[1])
    with pytest.raises(ModelError) as raised:
        call(model_for(server, timeout=0.1, mode="json_object"))
    assert raised.value.code == "provider_timeout" and len(server.requests) == 2


def test_without_usage_tokens_are_estimated(serve):
    server = serve(lambda m, p, b: completion('{"a": 4}'))
    _, receipt = call(model_for(server))
    assert receipt.tokens_estimated is True and receipt.input_tokens > 0 and receipt.output_tokens > 0


@pytest.mark.parametrize("status,message,code", [
    (401, "bad key", "provider_auth"), (403, "forbidden", "provider_auth"), (402, "pay", "out_of_credit"),
    (404, "no such model", "model_not_found"), (400, "messages must not be empty", "provider_rejected"),
    (500, "boom", "provider_failed"), (429, "slow down", "provider_rate_limited")])
def test_errors_become_model_error_codes(serve, status, message, code):
    server = serve(lambda m, p, b: error(status, message))
    with pytest.raises(ModelError) as raised:
        call(model_for(server))
    assert raised.value.code == code


def test_a_busy_provider_is_retried(serve):
    answers = iter([error(429, "slow down"), error(503, "busy"), completion('{"a": 5}')])
    server = serve(lambda m, p, b: next(answers))
    assert call(model_for(server))[0] == {"a": 5} and len(server.requests) == 3


def test_a_reply_without_a_json_object_is_invalid(serve):
    server = serve(lambda m, p, b: completion("I cannot do that."))
    with pytest.raises(ModelError) as raised:
        call(model_for(server))
    assert raised.value.code == "invalid_model_json"


def test_an_unreachable_provider_is_reported():
    model = OpenAICompatibleModel(base_url="http://127.0.0.1:9/v1", model="m", sleep=lambda s: None, timeout=2)
    with pytest.raises(ModelError) as raised:
        call(model)
    assert raised.value.code == "provider_unreachable"


def test_parse_json_object():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": {"b": 2}} Done.') == {"a": {"b": 2}}
    assert parse_json_object("[1, 2]") is None and parse_json_object("nothing") is None


def test_from_env_presets_and_errors():
    model = OpenAICompatibleModel.from_env({"DAOHARNESS_PROVIDER": "openai", "DAOHARNESS_MODEL": "m",
                                            "OPENAI_API_KEY": "k"})
    assert (model.base_url, model.model, model.api_key, model.provider) == ("https://api.openai.com/v1", "m", "k",
                                                                            "openai")
    gemini = OpenAICompatibleModel.from_env({"DAOHARNESS_PROVIDER": "gemini", "DAOHARNESS_MODEL": "g",
                                             "GEMINI_API_KEY": "gk", "DAOHARNESS_MAX_TOKENS": "4000"})
    assert gemini.base_url.endswith("/v1beta/openai") and gemini.api_key == "gk" and gemini.max_tokens == 4000
    assert gemini.mode == "json_object"
    local = OpenAICompatibleModel.from_env({"DAOHARNESS_PROVIDER": "ollama", "DAOHARNESS_MODEL": "llama"})
    assert local.base_url == "http://localhost:11434/v1" and local.api_key is None
    custom = OpenAICompatibleModel.from_env({"DAOHARNESS_BASE_URL": "https://gateway.example/v1/",
                                             "DAOHARNESS_MODEL": "x", "DAOHARNESS_API_KEY": "own"})
    assert (custom.base_url, custom.api_key, custom.provider) == ("https://gateway.example/v1", "own",
                                                                  "gateway.example")
    for environ, wanted in [({}, "DAOHARNESS_PROVIDER"), ({"DAOHARNESS_PROVIDER": "openai"}, "DAOHARNESS_MODEL"),
                            ({"DAOHARNESS_PROVIDER": "openai", "DAOHARNESS_MODEL": "m"}, "OPENAI_API_KEY"),
                            ({"DAOHARNESS_PROVIDER": "nope", "DAOHARNESS_MODEL": "m"}, "Unknown")]:
        with pytest.raises(ValueError, match=wanted):
            OpenAICompatibleModel.from_env(environ)


# --- replies as models really send them ----------------------------------------------------------------------------

# Wrapped in thinking, a code fence or prose, or with a comma left before a closing bracket: the whole object is there.
TIDY_REPLIES = [
    pytest.param('<think>The shape is {"a": ...}. Let me fill it in.</think>\n{"a": 1}', {"a": 1}, id="think-braces"),
    pytest.param('<think>{ draft }</think>\n```json\n{"a": 1}\n```', {"a": 1}, id="think-then-fence"),
    pytest.param('Reasoning with {braces} first.\n</think>\n{"a": 1}', {"a": 1}, id="closing-think-only"),
    pytest.param('```json\n{"a": 1}', {"a": 1}, id="fence-never-closed"),
    pytest.param('{"a": 1}\n```', {"a": 1}, id="fence-only-after"),
    pytest.param('```json\n```json\n{"a": 1}\n```\n```', {"a": 1}, id="fence-twice"),
    pytest.param('{"a": 1,}', {"a": 1}, id="trailing-comma"),
    pytest.param('{"a": 1, "b": [1, 2, 3,], "c": {"d": 4,\n},\n}', {"a": 1, "b": [1, 2, 3], "c": {"d": 4}},
                 id="trailing-commas-nested"),
    pytest.param('{"a": "x,}", "b": ["y, ]",]}', {"a": "x,}", "b": ["y, ]"]}, id="commas-in-strings-stay"),
    pytest.param('<think>try {"a": 0}</think>Here you go:\n```json\n{"a": 1, "b": [2,],}\n```\nHope that helps!',
                 {"a": 1, "b": [2]}, id="everything-at-once"),
]

# More than the one object, or something that is not strict JSON: never guessed at, never cut short.
REFUSED_REPLIES = [
    pytest.param('{"a": 1}\n{"a": 2}', id="two-objects"),
    pytest.param('```json\n{"a": 1}\n```\n```json\n{"a": 2}\n```', id="two-objects-in-two-fences"),
    pytest.param('{"a": 1} [1, 2]', id="object-then-list"),
    pytest.param('{"a": 1}}', id="stray-closing-brace"),
    pytest.param('[{"a": 1}]', id="list-holding-the-object"),
    pytest.param("{'a': 1}", id="single-quotes"),
    pytest.param("{a: 1}", id="unquoted-key"),
    pytest.param('{"a": 1 // one\n}', id="comment"),
    pytest.param('{"a": 1 "b": 2}', id="missing-comma"),
    pytest.param('{"a": [,1]}', id="comma-before-the-first-value"),
    pytest.param('{"a": [1,,2]}', id="doubled-comma"),
    pytest.param('{"a": [1, 2', id="cut-off"),
    pytest.param('<think>{"a": 1}', id="think-never-closed"),
]


@pytest.mark.parametrize("reply, expected", TIDY_REPLIES)
def test_a_wrapped_reply_gives_its_object(reply, expected):
    assert parse_json_object(reply) == expected


@pytest.mark.parametrize("reply", REFUSED_REPLIES)
def test_a_reply_with_more_than_the_object_or_broken_json_is_refused(reply):
    assert parse_json_object(reply) is None


def test_what_was_tidied_is_in_the_receipt(serve):
    server = serve(lambda m, p, b: completion('<think>x {y}</think>\n```json\n{"a": 1,}\n```'))
    content, receipt = call(model_for(server))
    assert content == {"a": 1}
    assert receipt.repairs == ("removed a <think> block", "removed a code fence", "removed a trailing comma")
    assert receipt.to_dict()["repairs"] == list(receipt.repairs)
    _, clean = call(model_for(serve(lambda m, p, b: completion('{"a": 1}'))))
    assert clean.repairs is None and "repairs" not in clean.to_dict()


def test_extra_data_after_the_object_is_an_error_that_says_so(serve):
    server = serve(lambda m, p, b: completion('{"a": 1}\n{"a": 2}'))
    with pytest.raises(ModelError) as raised:
        call(model_for(server))
    assert raised.value.code == "invalid_model_json"
    assert "extra data after the JSON object" in raised.value.detail


def test_a_messy_reply_still_builds_a_validated_workflow(serve):
    """Thinking with braces, a fence, a trailing comma and a capitalised step id: tidied, then checked as usual."""
    payload = as_model_payload(ACCOUNTING["spec"])
    payload["spec"]["steps"][0]["id"] = "Collect"
    reply = ("<think>The first step is {collect}, then the rest.</think>\n```json\n"
             + json.dumps(payload).replace('"declared_gaps": []}', '"declared_gaps": [],}') + "\n```")
    assert reply.endswith(',}\n```')
    server = serve(lambda m, p, b: completion(reply))
    ctx = ValidationContext(builtin_catalog(), lambda role: True)
    result = WorkflowCompiler(model_for(server)).compile(
        org_id="o", description="Intake.", ctx=ctx, primitive_refs=builtin_catalog().ids())
    assert [s.id for s in result.spec.steps][:2] == ["collect", "extract"]
    assert result.receipts[0]["repairs"] == ["removed a <think> block", "removed a code fence",
                                             "removed a trailing comma"]
    assert any("<think> block" in note for note in result.notes)
    assert any("lower-cased" in note for note in result.notes)


# --- failures a person can act on ---------------------------------------------------------------------------------

def test_a_request_the_provider_rejects_carries_its_status(serve):
    server = serve(lambda m, p, b: error(400, "messages must not be empty"))
    with pytest.raises(ModelError) as raised:
        call(model_for(server))
    assert (raised.value.code, raised.value.detail) == ("provider_rejected", "HTTP 400")
    assert "messages must not be empty" not in str(raised.value) + raised.value.detail


@pytest.mark.parametrize("finish_reason", ["length", "LENGTH", "max_tokens", "MAX_TOKENS"])
def test_a_reply_cut_off_at_the_token_limit_is_named_as_cut_off(serve, finish_reason):
    server = serve(lambda m, p, b: completion('{"a": [1, 2', finish_reason=finish_reason))
    with pytest.raises(ModelError) as raised:
        call(model_for(server, max_tokens=5))
    assert raised.value.code == "reply_truncated"


def test_a_reply_with_no_text_at_the_token_limit_is_cut_off_too(serve):
    # A model that spends every token thinking sends back nothing at all
    server = serve(lambda m, p, b: completion("", finish_reason="length"))
    with pytest.raises(ModelError) as raised:
        call(model_for(server))
    assert raised.value.code == "reply_truncated"


def test_a_complete_object_is_used_even_when_the_provider_says_length(serve):
    server = serve(lambda m, p, b: completion('{"a": 6}', finish_reason="length"))
    assert call(model_for(server))[0] == {"a": 6}


def test_bad_json_that_was_not_cut_off_stays_invalid_json_with_the_reason(serve):
    server = serve(lambda m, p, b: completion("{'a': 1}", finish_reason="stop"))
    with pytest.raises(ModelError) as raised:
        call(model_for(server))
    assert raised.value.code == "invalid_model_json" and raised.value.detail.startswith("syntax error")


@pytest.mark.parametrize("answer", [
    lambda: (200, {"Content-Type": "text/html"}, b"<html>Welcome</html>"),
    lambda: (200, {"Content-Type": "application/json"}, b"[1, 2]"),
    lambda: (200, {"Content-Type": "application/json"}, b'{"choices": []}'),
    lambda: (200, {"Content-Type": "application/json"}, b'{"choices": [{"message": {"content": 7}}]}'),
], ids=["html", "a-list", "no-choices", "no-text"])
def test_an_answer_that_is_not_a_chat_completion_points_at_the_address(serve, answer):
    server = serve(lambda m, p, b: answer())
    with pytest.raises(ModelError) as raised:
        call(model_for(server))
    assert raised.value.code == "provider_bad_reply"


def test_a_provider_that_does_not_answer_in_time_is_a_timeout_not_unreachable(serve):
    server = serve(lambda m, p, b: (time.sleep(0.6), completion('{"a": 1}'))[1])
    with pytest.raises(ModelError) as raised:
        call(model_for(server, timeout=0.1))
    assert raised.value.code == "provider_timeout"
