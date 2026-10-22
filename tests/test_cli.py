"""The terminal against the local server: the real CLI, the real routes and session checks, stand-in models."""
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

from daoharness.testing import FakeDecisionPort, FakeModel, as_model_payload  # noqa: E402
from daoharness_cli import main as cli  # noqa: E402
from daoharness_server.app import create_app, local_service  # noqa: E402
from daoharness_server.local_session import COOKIE, ensure_token  # noqa: E402
from industry_fixtures import fan_out_fake_outputs, recruiting_fan_out_sample, recruiting_fan_out_spec  # noqa: E402


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Keyboard:
    """A terminal for one test: isatty() is true and input() answers in order."""

    def __init__(self, answers):
        self.answers = list(answers)

    def isatty(self):
        return True

    def read(self):
        return ""

    def answer(self, prompt=""):
        return self.answers.pop(0)


@pytest.fixture
def local(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(cli, "HOME", home)
    monkeypatch.setattr(cli, "CONFIG", home / "config.json")
    monkeypatch.setattr(cli, "POLL_SECONDS", 0.05)
    for name in ("DAOHARNESS_URL", "DAOHARNESS_SESSION", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
    model = FakeModel(compile_outputs=[as_model_payload(recruiting_fan_out_spec())] * 5,
                      step_overrides=fan_out_fake_outputs())
    port = free_port()
    app = create_app(local_service(home, model, decisions=FakeDecisionPort()), token=ensure_token(home), port=port)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    url = f"http://127.0.0.1:{port}"

    def run(command, *rest):
        return cli.main([command, "--url", url, "--no-color", *rest])

    yield SimpleNamespace(url=url, home=home, run=run)
    server.should_exit = True
    thread.join(timeout=5)


def test_build_save_and_run_with_a_batch_review(local, tmp_path, capsys, monkeypatch):
    assert local.run("status") == 0
    assert "DaoHarness local server" in capsys.readouterr().out
    assert local.run("new", "Screen each applicant in the packet and draft a reply for the hiring lead", "--save") == 0
    out = capsys.readouterr().out
    assert "Saved as version 1" in out and "you approve" in out and "per applicant" in out

    packet = tmp_path / "input.json"
    packet.write_text(json.dumps(recruiting_fan_out_sample()))
    keyboard = Keyboard(["r 2", "e 1", ""])
    monkeypatch.setattr(sys, "stdin", keyboard)
    monkeypatch.setattr("builtins.input", keyboard.answer)
    edits = []

    def editor(ui, value):
        edits.append(value)
        changed = json.loads(json.dumps(value))
        draft = changed["step:outreach"]
        draft["body"] = draft["body"] + " (edited)"
        return changed

    monkeypatch.setattr(cli, "edit_json", editor)
    assert local.run("run", "--input", str(packet)) == 0
    out = capsys.readouterr().out
    assert "Input  from you" in out and "approves:" in out
    assert "✗ 2 rejected" in out and "✎ 1 edited" in out and "✓ 1 approved · ✗ 1 rejected · ✎ 1 edited" in out
    assert edits and set(edits[0]) == {"step:outreach"}
    assert "Usage" in out and "model call" in out and "cheaper" not in out
    assert keyboard.answers == []

    monkeypatch.setattr(sys, "stdin", Keyboard([]))
    assert local.run("approvals") == 0
    assert "Nothing is waiting for you" in capsys.readouterr().out
    assert local.run("ls") == 0
    assert "1 run, last complete" in capsys.readouterr().out


def test_a_run_needs_input_when_the_server_makes_no_examples(local, capsys):
    assert local.run("new", "Screen each applicant", "--save") == 0
    capsys.readouterr()
    assert local.run("run") == 1
    assert "Give the inputs with --input FILE" in capsys.readouterr().err


def request(url, method="GET", headers=None, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json",
                                                                         **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


def test_the_local_server_needs_its_session_the_header_and_this_machine(local):
    token = (local.home / "local-session").read_text()
    catalog = f"{local.url}/web/business-capabilities/catalog"
    assert request(catalog)[0] == 401
    assert request(catalog, headers={"Cookie": f"{COOKIE}=wrong"})[0] == 401
    assert request(catalog, headers={"Cookie": f"{COOKIE}={token}"}) == (200, {"creations": []})
    status, body = request(f"{local.url}/web/business-capabilities/creations", "POST",
                           {"Cookie": f"{COOKIE}={token}"}, {"action_id": "a1", "need": "x"})
    assert (status, body["detail"]["code"]) == (403, "WEB_CSRF_REQUIRED")
    for headers in ({"Host": "evil.example"}, {"Origin": "http://evil.example"}, {"Sec-Fetch-Site": "cross-site"}):
        status, body = request(catalog, headers={"Cookie": f"{COOKIE}={token}", **headers})
        assert (status, body["detail"]["code"]) == (403, "LOCAL_ORIGIN_REQUIRED")
    assert oct((local.home / "local-session").stat().st_mode & 0o777) == "0o600"


def test_a_saved_sign_in_wins_and_logout_returns_to_the_local_server(local, capsys, monkeypatch):
    token = (local.home / "local-session").read_text()
    monkeypatch.setenv("DAOHARNESS_SESSION", f"{COOKIE}={token}")
    assert cli.main(["login", "--url", local.url, "--from-env", "--no-color"]) == 0
    monkeypatch.delenv("DAOHARNESS_SESSION")
    assert cli.read_config() == {"url": local.url, "cookie_name": COOKIE, "session": token}
    (local.home / "local-session").unlink()
    assert cli.main(["status", "--no-color"]) == 0          # the saved sign-in for this address
    assert cli.main(["logout"]) == 0 and cli.read_config() == {}
    assert cli.main(["status", "--url", local.url]) == 1      # no saved sign-in and no session file
    assert "daoharness serve" in capsys.readouterr().err


def test_paths_under_home_are_shown_from_tilde(monkeypatch):
    monkeypatch.setenv("HOME", "/home/someone")
    assert cli.tilde("/home/someone/.daoharness/deliveries/x") == "~/.daoharness/deliveries/x"
    assert cli.tilde("/home/someone-else/x") == "/home/someone-else/x" and cli.tilde("/tmp/x") == "/tmp/x"


def test_usage_shows_calls_tokens_and_time(capsys):
    ui = cli.UI(SimpleNamespace(no_color=True, yes=False, full=False))
    cli.show_usage(ui, {"calls": 4, "input_tokens": 12000, "output_tokens": 900}, {"model_calls": 4,
                                                                                   "elapsed_seconds": 75})
    assert "4 model calls · 12,000 tokens in · 900 out · 1:15 start to finish" in capsys.readouterr().out
    cli.show_usage(ui, None, {"model_calls": 1})
    assert "1 model call" in capsys.readouterr().out


def test_error_messages_name_the_next_step():
    assert cli.is_local("http://127.0.0.1:8767") and cli.is_local("localhost:8767")
    assert not cli.is_local("https://hosted.example")
    assert "daoharness serve" in cli.explain(cli.ApiError(0, "network", "refused"), "127.0.0.1:8767")
    assert "Can't reach" in cli.explain(cli.ApiError(0, "network", "refused"), "hosted.example")
    assert "daoharness login" in cli.explain(cli.ApiError(401, "WEB_SESSION_REQUIRED", "x"), "hosted.example")
