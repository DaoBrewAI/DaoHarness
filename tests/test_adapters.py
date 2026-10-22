"""The ready-made model port over an OpenAI-compatible endpoint, against a local stand-in (no network beyond
127.0.0.1)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from daoharness.adapters import OpenAICompatibleModel, parse_json_object
from daoharness.ports import ModelError, ModelReceipt

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


def completion(content, **extra):
    return 200, {"Content-Type": "application/json"}, json.dumps(
        {"id": "chatcmpl-1", "model": "served-model", "choices": [{"message": {"role": "assistant", "content": content}}],
         **extra}).encode()


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
    assert '"quote":"hi"' in user["content"] and "JSON Schema" not in user["content"]


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
