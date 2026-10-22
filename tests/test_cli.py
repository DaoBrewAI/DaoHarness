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

    yield SimpleNamespace(url=url, home=home, run=run, model=model)
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


def test_a_build_the_engine_tidied_is_announced_with_the_workflow(local, capsys):
    payload = as_model_payload(recruiting_fan_out_spec())
    for step in payload["spec"]["steps"]:
        step["id"] = step["id"].title()
        step["for_each"] = step["for_each"] and step["for_each"].title()
    local.model.compile_outputs[:] = [payload]
    assert local.run("new", "Screen each applicant", "--save") == 0
    out = capsys.readouterr().out
    assert "Note: Step ids were lower-cased, with every reference to them: Collect to collect" in out
    assert "Saved as version 1" in out


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
    assert cli.explain(cli.ApiError(0, "network", "refused"), "127.0.0.1:8767") == (
        "The local server is not running. Start it in another terminal: daoharness serve")
    assert cli.explain(cli.ApiError(0, "network", "name not known"), "hosted.example") == (
        "Can't reach hosted.example (name not known). Check your internet connection and the address, then try "
        "again.")
    assert "daoharness login" in cli.explain(cli.ApiError(401, "WEB_SESSION_REQUIRED", "x"), "hosted.example")


def test_a_server_that_names_its_error_is_not_asked_again(monkeypatch):
    pauses, sent = [], []
    client = cli.Client("http://127.0.0.1:1", "s", pause=pauses.append)

    def refuse(method, path, body, *rest):
        sent.append(path)
        raise cli.ApiError(503, "provider_auth", "The model provider did not accept your key.", reported=True)

    monkeypatch.setattr(client, "_send", refuse)
    with pytest.raises(cli.ApiError) as raised:
        client.get("/x")
    assert raised.value.code == "provider_auth" and sent == ["/x"] and pauses == []


def test_a_gateway_that_said_nothing_useful_is_asked_again(monkeypatch):
    pauses, sent = [], []
    client = cli.Client("http://127.0.0.1:1", "s", pause=pauses.append)

    def busy(method, path, body, *rest):
        sent.append(path)
        raise cli.ApiError(503, "", "HTTP 503")

    monkeypatch.setattr(client, "_send", busy)
    with pytest.raises(cli.ApiError):
        client.get("/x")
    assert len(sent) == 5 and pauses == [1.0, 2.0, 4.0, 8.0]


def test_only_the_servers_own_error_format_counts_as_naming_an_error():
    named = cli.api_error(503, {"detail": {"code": "provider_failed", "message": "m"}})
    assert named.reported and (named.code, named.message) == ("provider_failed", "m")
    for payload in (None, {}, {"detail": "Service Unavailable"}, {"detail": [{"msg": "x"}]}):
        assert not cli.api_error(503, payload).reported


def test_a_model_failure_reaches_the_terminal_as_a_sentence_and_at_once(local, capsys):
    local.model.fail_operations.add("compile")
    started = time.monotonic()
    assert local.run("new", "Screen each applicant") == 1
    err = capsys.readouterr().err
    assert err.strip().endswith("The model provider had an error on its side. Try again in a minute. (provider_failed)")
    assert "unavailable" not in err and time.monotonic() - started < 5


def test_a_workflow_that_fails_validation_reaches_the_terminal_with_its_step_ids(local, capsys):
    bad = as_model_payload(recruiting_fan_out_spec())
    bad["spec"]["steps"][2]["inputs"] = ["step:nowhere"]
    local.model.compile_outputs[:] = [bad, json.loads(json.dumps(bad))]
    assert local.run("new", "Screen each applicant") == 1
    err = capsys.readouterr().err
    assert "did not pass validation, even on a second try: step 'extract' reads step 'nowhere'" in err
    assert err.strip().endswith("(compile_invalid)")


def test_a_decision_the_model_could_not_answer_says_so_instead_of_none(capsys):
    ui = cli.UI(SimpleNamespace(no_color=True, yes=True, full=False))
    posted = []
    client = SimpleNamespace(post=lambda path, body: (200, posted.append(body) or {}))
    item = {"id": "docs", "question": "Is it there?", "options": ["present", "missing"],
            "candidates": [{"rung": "model", "choice": None, "confidence": None},
                           {"rung": "strong_model", "choice": "missing", "confidence": 0.6}]}
    cli.review(ui, client, {"approval_id": "a", "content_digest": "d", "items": [item]})
    out = capsys.readouterr().out
    assert "model gave no usable answer" in out and "stronger model said “missing” (60% sure)" in out
    assert "None" not in out and posted[0]["choices"] == {"docs": "missing"}


# --- a long local build ------------------------------------------------------------------------------------------

class RecordingLive:
    """Stands in for the status line: it keeps what was set and counts the clears."""

    def __init__(self, tty=True):
        self.tty, self.texts, self.cleared = tty, [], 0

    def set(self, text):
        self.texts.append(text)

    def clear(self):
        self.cleared += 1

    def say(self, line=""):
        print(line)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _ui(tty=True):
    ui = cli.UI(SimpleNamespace(no_color=True, yes=False, full=False))
    ui.live = RecordingLive(tty)
    return ui


def test_the_waiting_line_counts_the_seconds_and_after_a_while_says_a_local_model_is_slow():
    assert cli.waiting_text("Building the workflow", 0) == "Building the workflow · 0:00"
    assert cli.waiting_text("Building the workflow", 19.9) == "Building the workflow · 0:19"
    assert cli.waiting_text("Building the workflow", 75) == (
        "Building the workflow · 1:15 · a local model can take a few minutes")


def test_the_waiting_line_updates_while_the_call_is_open_and_goes_when_it_ends(monkeypatch):
    clock, ui = Clock(), _ui()
    monkeypatch.setattr(cli.time, "monotonic", clock)
    with cli.waiting(ui, "Building the workflow"):
        [line] = ui.live.texts
        assert line() == "Building the workflow · 0:00"
        clock.now += 42
        assert line() == "Building the workflow · 0:42 · a local model can take a few minutes"
        assert ui.live.cleared == 0
    assert ui.live.cleared == 1


def test_the_waiting_line_goes_when_the_call_fails_too():
    ui = _ui()
    with pytest.raises(RuntimeError):
        with cli.waiting(ui, "Building the workflow"):
            raise RuntimeError("boom")
    assert ui.live.cleared == 1


def test_without_a_terminal_the_wait_is_one_plain_line(capsys):
    ui = _ui(tty=False)
    with cli.waiting(ui, "Building the workflow"):
        pass
    assert ui.live.texts == [] and capsys.readouterr().out == "Building the workflow…\n"


def test_hosted_builds_are_posted_as_before_with_no_waiting_line():
    ui, posted = _ui(), []

    class Hosted:
        url = "https://hosted.example"

        def post(self, path, body, **kw):
            posted.append((path, body, kw))
            return 202, {"status": "building"}

    assert cli.start_build(ui, Hosted(), "/creations", {"need": "x"}, "Building the workflow") == (
        202, {"status": "building"})
    assert posted == [("/creations", {"need": "x"}, {})] and ui.live.texts == []


@pytest.fixture
def slow_server(local, monkeypatch):
    """The real terminal and local server, but a creations call that takes `needs` seconds to answer, simulated: it
    times out unless the terminal is willing to wait that long. No real waiting, and no pause between retries."""
    calls = []
    real_urlopen, real_connect = cli.urllib.request.urlopen, cli.connect

    def urlopen(request, timeout=None, context=None):
        if request.full_url.endswith("/creations"):
            calls.append(timeout)
            if timeout < local.needs:
                raise TimeoutError("timed out")
        return real_urlopen(request, timeout=timeout, context=context)

    def connect(ui, args):
        client = real_connect(ui, args)
        client.pause = lambda seconds: None
        return client

    local.needs, local.calls = 100, calls
    monkeypatch.setattr(cli.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(cli, "connect", connect)
    return local


def test_a_local_build_that_takes_minutes_is_waited_for(slow_server, capsys):
    assert slow_server.run("new", "Screen each applicant", "--save") == 0
    out = capsys.readouterr().out
    assert "Building the workflow" in out and "Saved as version 1" in out
    assert slow_server.calls == [cli.BUILD_LIMIT_SECONDS]


def test_a_local_build_that_never_finishes_ends_with_a_clear_message_after_one_wait(slow_server, capsys):
    slow_server.needs = cli.BUILD_LIMIT_SECONDS + 1
    assert slow_server.run("new", "Screen each applicant") == 1
    err = capsys.readouterr().err
    assert err.strip() == (
        "✗ The model is still working after 15 minutes, so the terminal stopped waiting. The local server keeps "
        "going: look for the workflow in a few minutes with daoharness ls, before you ask again. If this keeps "
        "happening, use a faster model.")
    assert slow_server.calls == [cli.BUILD_LIMIT_SECONDS]


def test_a_real_slow_server_is_waited_for_past_the_ordinary_timeout(local, capsys, monkeypatch):
    generate = local.model.generate

    def slow(**kw):
        time.sleep(1.2)
        return generate(**kw)

    local.model.generate = slow
    monkeypatch.setattr(cli, "REQUEST_TIMEOUT", 0.4)
    monkeypatch.setattr(cli, "BUILD_LIMIT_SECONDS", 20)
    started = time.monotonic()
    assert local.run("new", "Screen each applicant", "--save") == 0
    assert "Saved as version 1" in capsys.readouterr().out and time.monotonic() - started >= 1.2


def test_a_real_slow_server_past_the_limit_is_given_up_on_once(local, capsys, monkeypatch):
    generate, release, asked = local.model.generate, threading.Event(), []

    def stuck(**kw):
        asked.append(kw["operation"])
        release.wait(20)     # a model that does not answer until the test lets go of it
        return generate(**kw)

    local.model.generate = stuck
    monkeypatch.setattr(cli, "REQUEST_TIMEOUT", 0.2)
    monkeypatch.setattr(cli, "BUILD_LIMIT_SECONDS", 0.6)
    try:
        started = time.monotonic()
        assert local.run("new", "Screen each applicant") == 1
        assert "still working after" in capsys.readouterr().err and time.monotonic() - started < 10
        assert asked == ["compile"]      # asked once, and not again after the terminal gave up
    finally:
        release.set()


def test_a_timeout_is_retried_at_the_ordinary_limit_but_not_after_a_long_wait(monkeypatch):
    pauses, sent = [], []
    client = cli.Client("http://127.0.0.1:1", "s", pause=pauses.append)

    def too_slow(method, path, body, timeout):
        sent.append(timeout)
        raise cli.ApiError(0, "timeout", "timed out")

    monkeypatch.setattr(client, "_send", too_slow)
    with pytest.raises(cli.ApiError):
        client.get("/x")
    assert sent == [cli.REQUEST_TIMEOUT] * 5
    sent.clear()
    with pytest.raises(cli.ApiError) as raised:
        client.post("/x", {}, timeout=cli.BUILD_LIMIT_SECONDS)
    assert sent == [cli.BUILD_LIMIT_SECONDS] and raised.value.code == "timeout"


def test_the_curl_fallback_waits_as_long_and_names_a_timeout(monkeypatch):
    client = cli.Client("http://127.0.0.1:1", "s")
    commands = []

    def run(command, **kw):
        commands.append(command)
        return SimpleNamespace(returncode=28, stdout=b"", stderr=b"Operation timed out")

    monkeypatch.setattr(cli.subprocess, "run", run)
    with pytest.raises(cli.ApiError) as raised:
        client._send_curl("POST", "/x", {}, 900)
    assert commands[0][commands[0].index("--max-time") + 1] == "900"
    assert (raised.value.status, raised.value.code) == (0, "timeout")


def test_a_timeout_reads_as_slow_not_as_a_server_that_is_not_running():
    assert cli.explain(cli.ApiError(0, "timeout", "timed out"), "127.0.0.1:8767") == (
        "The local server did not answer in time. Try again in a minute.")
    assert cli.explain(cli.ApiError(0, "timeout", "timed out"), "hosted.example") == (
        "hosted.example did not answer in time. Try again in a minute.")
