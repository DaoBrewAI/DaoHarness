#!/usr/bin/env python3
"""daoharness: build, run and approve workflows from your terminal.

It talks to the local server (daoharness serve, on 127.0.0.1, with your own model) or to DaoHarness hosted
(daoharness login --url ...). Same commands, same routes; what you see depends on what the server does.

  daoharness serve                  start the local server with your model (pip install "daoharness[server]")
  daoharness new ["one sentence"]   build a workflow from one sentence, then save it
  daoharness run [WORKFLOW]         run a saved workflow on your input (--input FILE, --text, or paste it)
  daoharness approvals              approve, reject or edit what is waiting for you, then follow the run
  daoharness watch [WORKFLOW]       follow a workflow's latest run
  daoharness ls | show [WORKFLOW] | status
  daoharness login --url URL        use DaoHarness hosted instead (your session stays on this computer)
  daoharness logout                 back to the local server

WORKFLOW is part of a workflow's name or id; leave it out for the most recent one.
Standard library only; Python 3.9 or newer.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import shlex
import shutil
import ssl
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta
from getpass import getpass
from pathlib import Path

VERSION = "0.1.0"
DEFAULT_URL = "http://127.0.0.1:8767"   # the local server (daoharness serve)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
API = "/web/business-capabilities"
COOKIE = "daobrew_web_session"
HOME = Path(os.environ.get("DAOHARNESS_HOME") or Path.home() / ".daoharness")
CONFIG = HOME / "config.json"
POLL_SECONDS = 1.5
BUILD_LIMIT_SECONDS = 15 * 60
DECIDING = {"checklist_match", "screen_score", "classify_route"}
SETTLED = {"complete", "rejected", "failed", "needs_attention", "needs_access", "needs_capability", "interrupted",
           "needs_revision", "needs_clarification"}
RUNGS = {"model": "model", "strong_model": "stronger model", "human": "you"}   # any other rung: a decision engine
SOURCE_LABELS = {"verified": "quote found on the page", "grounded": "Google Search", "unchecked": "page not checked"}
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Stop(Exception):
    """A message for the person; the command ends with exit status 1."""


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


# -- text -----------------------------------------------------------------------------------------------------------

class Style:
    def __init__(self, on: bool) -> None:
        self.on = on

    def _c(self, code: str, text: object) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else str(text)

    def bold(self, text: object) -> str:
        return self._c("1", text)

    def dim(self, text: object) -> str:
        return self._c("2", text)

    def blue(self, text: object) -> str:
        return self._c("38;5;63", text)

    def yellow(self, text: object) -> str:
        return self._c("38;5;178", text)

    def green(self, text: object) -> str:
        return self._c("38;5;35", text)

    def red(self, text: object) -> str:
        return self._c("38;5;167", text)

    def link(self, url: str, text: str) -> str:
        """A clickable link where the terminal supports it (OSC 8); plain text elsewhere."""
        return f"\033]8;;{url}\033\\{text}\033]8;;\033\\" if self.on else text


def width() -> int:
    return max(60, min(shutil.get_terminal_size((100, 24)).columns, 110))


def clip(text: object, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"


def first_line(text: object, limit: int) -> str:
    return clip(str(text).strip().split("\n")[0], limit)


def wrap(text: object, limit: int, indent: str = "  ", hang: str = "") -> list:
    lines = []
    for para in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(para, max(20, limit - len(indent)), subsequent_indent=hang) or [""])
    return [indent + line for line in lines]


def humanize(name: object) -> str:
    name = str(name).split(":", 1)[-1].replace("_", " ").replace("-", " ").strip()
    return name[:1].upper() + name[1:]


def clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def tilde(path: str) -> str:
    """A path under the home folder written from ~, as people type it."""
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if home not in ("", "~", os.sep) and path.startswith(home + os.sep) else path


def plural(noun: str, n: int) -> str:
    return noun if n == 1 or noun.endswith("s") else noun + "s"


def parse_time(value: object):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def new_action(kind: str) -> str:
    """Every write carries a fresh action id: the server applies an action once, however often it arrives."""
    return f"cli-{kind}-{uuid.uuid4().hex}"


class Live:
    """One status line at the bottom that redraws in place; everything else prints above it."""

    def __init__(self, style: Style) -> None:
        self.s = style
        self.tty = sys.stdout.isatty()
        self.text = None
        self.frame = 0
        self.lock = threading.Lock()
        self.thread = None
        self.cursor_hidden = False

    def _draw(self) -> None:
        text = self.text() if callable(self.text) else self.text
        sys.stdout.write("\r\033[K  " + self.s.blue(SPIN[self.frame % len(SPIN)]) + " " + clip(text, width() - 6))
        sys.stdout.flush()

    def _loop(self) -> None:
        while True:
            time.sleep(0.1)
            with self.lock:
                if self.text is not None:
                    self.frame += 1
                    self._draw()

    def set(self, text) -> None:
        if not self.tty:
            return
        with self.lock:
            if not self.cursor_hidden:
                sys.stdout.write("\033[?25l")
                self.cursor_hidden = True
            self.text = text
            self._draw()
        if self.thread is None:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def clear(self) -> None:
        with self.lock:
            if self.text is not None:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            self.text = None

    def say(self, line: object = "") -> None:
        with self.lock:
            if self.text is not None:
                sys.stdout.write("\r\033[K")
            sys.stdout.write(f"{line}\n")
            if self.text is not None:
                self._draw()
            sys.stdout.flush()

    def restore(self) -> None:
        self.clear()
        with self.lock:
            if self.cursor_hidden:
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
                self.cursor_hidden = False


class UI:
    def __init__(self, args: argparse.Namespace) -> None:
        colour = sys.stdout.isatty() and not getattr(args, "no_color", False) and not os.environ.get("NO_COLOR")
        self.s = Style(colour)
        self.live = Live(self.s)
        self.yes = bool(getattr(args, "yes", False))
        self.full = bool(getattr(args, "full", False))
        self.interactive = sys.stdin.isatty()

    def p(self, line: object = "") -> None:
        self.live.say(line)

    def prompt(self, text: str) -> str:
        self.live.restore()
        return input(text)

    def ask(self, question: str) -> bool:
        """Yes or no, yes by default. --yes answers yes; without a keyboard nothing is decided."""
        if self.yes:
            return True
        if not self.interactive:
            raise Stop("Waiting for your decision. Run this in a terminal, or add --yes.")
        return self.prompt(question).strip().lower() not in ("n", "no", "否", "不")


# -- the server -----------------------------------------------------------------------------------------------------

def load(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8") or "null")
    except (ValueError, UnicodeDecodeError):
        return None


def api_error(status: int, payload) -> ApiError:
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        return ApiError(status, str(detail.get("code") or ""), str(detail.get("message") or detail.get("code") or status))
    if isinstance(detail, list):  # request validation
        return ApiError(status, "invalid_request",
                        "; ".join(str(d.get("msg")) for d in detail if isinstance(d, dict)) or "Invalid request.")
    if isinstance(detail, str):
        return ApiError(status, detail, detail)
    return ApiError(status, "", f"HTTP {status}")


def ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    try:
        import certifi  # optional: some Pythons on macOS ship without certificates
        context.load_verify_locations(certifi.where())
    except Exception:
        pass
    return context


class Client:
    def __init__(self, url: str, session: str, cookie_name: str = COOKIE, pause=time.sleep) -> None:
        self.url = url.rstrip("/")
        self.host = urllib.parse.urlsplit(self.url).netloc
        self.session = session
        self.cookie_name = cookie_name
        self.pause = pause
        self.context = ssl_context()
        self.curl = False

    def get(self, path: str):
        return self.request("GET", path)

    def post(self, path: str, body: dict):
        return self.request("POST", path, body)

    def request(self, method: str, path: str, body: dict | None = None):
        """(status, payload). A dropped connection or a busy server is retried; writes are safe to repeat because
        each carries its action id."""
        delay = 1.0
        for attempt in range(5):
            try:
                return self._send(method, path, body)
            except ApiError as err:
                if err.status not in (0, 502, 503, 504) or attempt == 4:
                    raise
            self.pause(delay)
            delay = min(delay * 2, 8.0)
        raise AssertionError("unreachable")

    def _headers(self, body) -> dict:
        headers = {"Accept": "application/json", "Cookie": f"{self.cookie_name}={self.session}",
                   "X-DaoBrew-Web": "1", "User-Agent": f"daoharness-cli/{VERSION}"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        return headers

    def _send(self, method: str, path: str, body):
        if self.curl:
            return self._send_curl(method, path, body)
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.url + path, data=data, method=method, headers=self._headers(body))
        try:
            with urllib.request.urlopen(request, timeout=60, context=self.context) as response:
                return response.status, load(response.read())
        except urllib.error.HTTPError as exc:
            raise api_error(exc.code, load(exc.read())) from None
        except urllib.error.URLError as exc:
            if "CERTIFICATE_VERIFY_FAILED" in str(exc.reason) and shutil.which("curl"):
                self.curl = True  # this Python has no certificates; curl uses the system's
                return self._send_curl(method, path, body)
            raise ApiError(0, "network", str(exc.reason)) from None
        except OSError as exc:
            raise ApiError(0, "network", str(exc) or type(exc).__name__) from None

    def _send_curl(self, method: str, path: str, body):
        fd, header_file = tempfile.mkstemp(prefix="daoharness-")  # 0600: the session stays out of the process list
        try:
            with os.fdopen(fd, "w") as handle:
                for key, value in self._headers(body).items():
                    handle.write(f"{key}: {value}\n")
            command = ["curl", "-sS", "-X", method, "--max-time", "60", "-H", f"@{header_file}",
                       "-w", "\n%{http_code}", self.url + path]
            if body is not None:
                command += ["--data-binary", "@-"]
            done = subprocess.run(command, input=None if body is None else json.dumps(body).encode("utf-8"),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        finally:
            os.unlink(header_file)
        if done.returncode != 0:
            raise ApiError(0, "network", done.stderr.decode("utf-8", "replace").strip() or "curl failed")
        raw, _, code = done.stdout.rpartition(b"\n")
        status = int(code or 0)
        if status >= 400:
            raise api_error(status, load(raw))
        return status, load(raw)


def is_local(url: str) -> bool:
    return (urllib.parse.urlsplit(url if "//" in url else f"//{url}").hostname or "") in LOCAL_HOSTS


def explain(err: ApiError, host: str) -> str:
    if err.status == 0 and is_local(host):
        return "The local server is not running. Start it in another terminal: daoharness serve"
    if err.status == 0:
        return f"Can't reach {host}: {err.message}"
    if err.status == 401 and is_local(host):
        return "The local server did not accept this session. Restart it: daoharness serve"
    if err.status == 401:
        return "Your session is missing or has expired. Sign in to the app again, then run: daoharness login"
    if err.code in ("SEED_ORIGIN_REQUIRED", "LOCAL_ORIGIN_REQUIRED"):
        return f"The server refused this address ({host}). Use its exact address with --url."
    if err.code == "ACCOUNT_SCOPE_MISMATCH":
        return "Your sign-in has no workspace on the server yet."
    if err.status == 404 and err.code in ("", "Not Found", "SEED_ROUTE_NOT_AVAILABLE"):
        return "This server does not offer that."
    return f"{err.message} ({err.code or err.status})"


# -- sign-in --------------------------------------------------------------------------------------------------------

def read_config() -> dict:
    try:
        return json.loads(CONFIG.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def write_config(config: dict) -> None:
    HOME.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = CONFIG.with_suffix(".tmp")
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(config, handle)
    os.chmod(str(temporary), 0o600)
    os.replace(str(temporary), str(CONFIG))


def parse_session(text: str) -> tuple:
    """(cookie name, value) from the bare value, name=value, or a whole Cookie header."""
    text = text.strip().strip("'\"")
    if text.lower().startswith("cookie:"):
        text = text[len("cookie:"):]
    for part in text.split(";"):
        name, sep, value = part.strip().partition("=")
        if sep and name.startswith(COOKIE):
            return name, value.strip().strip("'\"")
    return COOKIE, text


def expires(token: str):
    try:
        part = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        return datetime.fromtimestamp(int(claims["exp"]))
    except Exception:
        return None


def target(args: argparse.Namespace, config: dict) -> str:
    return (args.url or os.environ.get("DAOHARNESS_URL") or config.get("url") or DEFAULT_URL).rstrip("/")


def connect(ui: UI, args: argparse.Namespace) -> Client:
    """The local server's session file for 127.0.0.1, or the session saved by login for that address."""
    config = read_config()
    url = target(args, config)
    session = os.environ.get("DAOHARNESS_SESSION") or (
        config.get("session") if (config.get("url") or "").rstrip("/") == url else None)
    if not session and is_local(url):
        try:
            session = (HOME / "local-session").read_text("utf-8").strip()
        except OSError:
            raise Stop("The local server has not started yet. Start it in another terminal: daoharness serve") from None
    if not session:
        raise Stop(f"Not signed in to {url}. Run: daoharness login --url {url}")
    name, value = parse_session(session)
    return Client(url, value, config.get("cookie_name") or name, pause=time.sleep)


def cmd_logout(ui: UI, args: argparse.Namespace) -> None:
    config = read_config()
    if config:
        write_config({})
    ui.p(ui.s.green("✓ ") + f"Signed out. Commands now go to the local server ({DEFAULT_URL}).")


def cmd_login(ui: UI, args: argparse.Namespace) -> None:
    s, config = ui.s, read_config()
    url = target(args, config)
    if is_local(url) and not args.from_env:
        ui.p(s.dim("The open-source local server needs no sign-in: the terminal reads its session file."))
    ui.p(s.bold("Sign in to DaoHarness hosted"))
    ui.p(f"  1. Open {url} in Chrome and sign in.")
    ui.p("  2. Open developer tools (⌥⌘I) → Application → Cookies → that address.")
    ui.p(f"  3. Copy the Value of {COOKIE}.")
    ui.p(s.dim(f"  It is saved only on this computer, in {CONFIG} (readable only by you)."))
    raw = os.environ.get("DAOHARNESS_SESSION", "") if args.from_env else getpass("  Paste it here (hidden): ")
    name, value = parse_session(raw)
    if not value:
        raise Stop("Nothing was pasted.")
    client = Client(url, value, name)
    client.get(f"{API}/catalog")
    write_config({"url": url, "cookie_name": name, "session": value})
    when = expires(value)
    ui.p(s.green("✓ ") + "Signed in" + (f" · valid until {when:%a %b %d, %H:%M}" if when else ""))


def cmd_status(ui: UI, client: Client, args: argparse.Namespace) -> None:
    s, local = ui.s, is_local(client.url)
    ui.p(s.bold("DaoHarness " + ("local server  " if local else "hosted  ")) + s.dim(client.url))
    _, catalog = client.get(f"{API}/catalog")
    when = None if local else expires(client.session)
    ui.p(s.green("✓ ") + ("Connected" if local else "Signed in")
         + (f" · valid until {when:%a %b %d, %H:%M}" if when else ""))
    if when is not None and when - datetime.now() < timedelta(hours=12):
        ui.p(s.yellow("! ") + "Your session ends soon. Sign in to the app again, then run: daoharness login")
    _, listing = client.get(f"{API}/approvals")
    count = len((catalog or {}).get("creations") or [])
    waiting = len((listing or {}).get("approvals") or [])
    ui.p(s.green("✓ ") + f"Workflows on · {count} workflow{'s' if count != 1 else ''} · "
         f"{waiting} waiting for you")


# -- workflows ------------------------------------------------------------------------------------------------------

def settle(ui: UI, client: Client, status: int, payload, label: str):
    """A queued build (202) is polled until the worker has run it; anything else is already the result."""
    if not (status == 202 and isinstance(payload, dict) and payload.get("status") == "building"):
        return payload
    build = payload.get("build") or {}
    kind, action, iid = build.get("kind"), build.get("action_id"), build.get("instance_id")
    path = f"{API}/builds/create/{action}" if kind == "create" else f"{API}/{iid}/builds/{kind}/{action}"
    started = time.monotonic()
    ui.live.set(lambda: f"{label} · {clock(time.monotonic() - started)}")
    while True:
        time.sleep(2.0)
        _, state = client.get(path)
        if isinstance(state, dict) and state.get("finalized"):
            ui.live.clear()
            return state.get("result")
        if time.monotonic() - started > BUILD_LIMIT_SECONDS:
            raise Stop("This is taking longer than expected. Check again in a minute; asking again never builds it twice.")


def item_noun(steps: list):
    for step in steps:
        if step.get("primitive") == "split_items" and (step.get("config") or {}).get("item_noun"):
            return str(step["config"]["item_noun"])
    return None


def step_tag(step: dict, noun) -> str:
    """Who acts, and how often."""
    primitive = step.get("primitive")
    if step.get("approval") or primitive == "human_approval":
        tag = "you approve"
    elif primitive in DECIDING:
        tag = "asks if unsure"
    elif primitive == "draft_message":
        tag = "draft only"
    else:
        tag = "automatic"
    if step.get("for_each"):
        tag += f" · per {noun or 'item'}"
    if "web_search" in (step.get("required_sources") or []):
        tag += " · web"
    return tag


def show_workflow(ui: UI, workflow, heading: str = "Workflow") -> None:
    s = ui.s
    workflow = workflow or {}
    spec, metrics = workflow.get("spec") or {}, workflow.get("metrics") or {}
    steps = spec.get("steps") or []
    noun = item_noun(steps)
    facts = [f"{metrics.get('total_steps', len(steps))} steps"]
    if metrics.get("automation_ratio") is not None:
        facts.append(f"{round(metrics['automation_ratio'] * 100)}% automatic")
    if metrics.get("fan_out"):
        facts.append(f"runs once per {noun or 'item'}")
    ui.p(s.bold(f"{heading}  {spec.get('domain') or 'Untitled'}") + s.dim("   " + " · ".join(facts)))
    if spec.get("purpose"):
        for line in wrap(spec["purpose"], width(), "  ")[:2]:
            ui.p(s.dim(line))
    ui.p()
    tags = [step_tag(step, noun) for step in steps]
    role_w = min(max([len(step.get("role") or "") for step in steps] or [8]), 22)
    tag_w = max([len(tag) for tag in tags] or [9])
    text_w = max(24, width() - 7 - role_w - 2 - tag_w - 2)
    for n, (step, tag) in enumerate(zip(steps, tags), 1):
        colour = s.yellow if tag.startswith("you approve") else s.blue
        ui.p(f"  {n:>2}  {clip(step.get('role') or '', role_w):<{role_w}}  "
             f"{first_line(step.get('instructions') or '', text_w):<{text_w}}  {colour(tag)}")
    ui.p()
    needs = [humanize(r.get("name")) for r in spec.get("input_requirements") or [] if r.get("required")]
    if needs:
        ui.p(s.dim(clip("  Needs     " + ", ".join(needs), width())))
    if spec.get("deliverables"):
        ui.p(s.dim(clip("  Delivers  " + "; ".join(str(d) for d in spec["deliverables"]), width())))
    for gap in workflow.get("gaps") or []:
        ui.p(s.yellow("  ! ") + clip(f"{gap.get('step_id')}: {gap.get('detail')}", width() - 4))


def recency(snap: dict) -> str:
    creation = snap.get("creation") or {}
    stamps = [creation.get("created_at") or ""] + [r.get("started_at") or "" for r in snap.get("runs") or []]
    return max(stamps)


def workflow_name(snap: dict) -> str:
    creation = snap.get("creation") or {}
    return ((snap.get("family") or {}).get("name")
            or ((creation.get("workflow") or {}).get("spec") or {}).get("domain")
            or (creation.get("method") or {}).get("domain")
            or clip(creation.get("need") or "Untitled", 48))


def instance_id(snap: dict) -> str:
    return snap["creation"]["instance"]["instance_id"]


def catalog(client: Client) -> dict:
    _, found = client.get(f"{API}/catalog")
    return found or {}


def resolve(client: Client, ref, *, saved: bool = False) -> dict:
    snaps = sorted(catalog(client).get("creations") or [], key=recency, reverse=True)
    if saved:
        snaps = [s for s in snaps if (s.get("creation") or {}).get("status") == "confirmed"] or snaps
    if not snaps:
        raise Stop('No workflows yet. Build one: daoharness new "…"')
    if not ref:
        return snaps[0]
    wanted = ref.lower()
    for snap in snaps:
        if instance_id(snap).startswith(wanted):
            return snap
    for snap in snaps:
        if wanted in workflow_name(snap).lower() or wanted in ((snap.get("creation") or {}).get("need") or "").lower():
            return snap
    raise Stop(f"No workflow matches “{ref}”. See: daoharness ls")


def spec_for(client: Client, iid: str, version) -> dict:
    try:
        _, found = client.get(f"{API}/{iid}/versions")
        for entry in (found or {}).get("versions") or []:
            if entry.get("version") == version and (entry.get("workflow") or {}).get("spec"):
                return entry["workflow"]["spec"]
    except ApiError:
        pass
    _, snap = client.get(f"{API}/{iid}")
    return (((snap or {}).get("creation") or {}).get("workflow") or {}).get("spec") or {}


def cmd_ls(ui: UI, client: Client, args: argparse.Namespace) -> None:
    s = ui.s
    found = catalog(client)
    snaps = sorted(found.get("creations") or [], key=recency, reverse=True)
    building = found.get("building") or []
    if not snaps and not building:
        ui.p('No workflows yet. Build one: daoharness new "…"')
        return
    ui.p(s.bold("Workflows"))
    for snap in snaps:
        creation = snap.get("creation") or {}
        metrics = (creation.get("workflow") or {}).get("metrics") or {}
        runs = snap.get("runs") or []
        facts = [f"v{creation.get('version')}" if creation.get("status") == "confirmed" else (creation.get("status") or "draft")]
        if metrics.get("total_steps"):
            facts.append(f"{metrics['total_steps']} steps")
        if metrics.get("automation_ratio") is not None:
            facts.append(f"{round(metrics['automation_ratio'] * 100)}% automatic")
        if runs:
            facts.append(f"{len(runs)} run{'s' if len(runs) != 1 else ''}, last {str(runs[0].get('state')).replace('_', ' ')}")
        ui.p(f"  {s.dim(instance_id(snap)[:8])}  {clip(workflow_name(snap), 42):<42}  {s.dim(' · '.join(facts))}")
    for entry in building:
        ui.p(f"  {s.blue('building')}  {clip(entry.get('need') or '', 50):<50}  {s.dim(clock(entry.get('elapsed_seconds') or 0))}")
    _, listing = client.get(f"{API}/approvals")
    waiting = len((listing or {}).get("approvals") or [])
    if waiting:
        ui.p()
        ui.p(s.yellow("⏸ ") + f"{waiting} waiting for you" + s.dim("  →  daoharness approvals"))


def banner(ui: UI, client: Client) -> None:
    ui.p(ui.s.bold("DaoHarness") + ui.s.dim(" · local · your model" if is_local(client.url) else " · hosted"))
    ui.p()


def cmd_show(ui: UI, client: Client, args: argparse.Namespace) -> None:
    snap = resolve(client, args.workflow)
    show_workflow(ui, (snap.get("creation") or {}).get("workflow"))


def cmd_new(ui: UI, client: Client, args: argparse.Namespace) -> None:
    s = ui.s
    need = " ".join(args.sentence).strip()
    banner(ui, client)
    if need:
        ui.p(s.bold("You type"))
        for line in wrap(f"“{need}”", width(), "  "):
            ui.p(line)
    elif ui.interactive:
        ui.p(s.bold("Describe a job you repeat, in one sentence"))
        need = ui.prompt("  › ").strip()
    if not need:
        raise Stop('Describe the job: daoharness new "Every Monday, …"')
    ui.p()
    started = time.monotonic()
    status, payload = client.post(f"{API}/creations", {"action_id": new_action("create"), "need": need,
                                                       "source": "workflows"})
    creation = settle(ui, client, status, payload, "Building the workflow")
    while (creation or {}).get("status") == "needs_clarification":
        if ui.yes or not ui.interactive:
            raise Stop(f"The harness asks: {creation.get('question')} Answer it in a terminal, or in the app.")
        for line in wrap(creation.get("question") or "Can you say a little more?", width(), "  "):
            ui.p(s.yellow(line))
        answer = ui.prompt("  › ").strip()
        iid = creation["instance"]["instance_id"]
        status, payload = client.post(f"{API}/{iid}/brainstorm", {
            "action_id": new_action("answer"), "draft_id": creation.get("draft_id") or creation.get("latest_draft_id"),
            "answer": answer})
        creation = settle(ui, client, status, payload, "Updating the workflow")
    if (creation or {}).get("status") != "draft":
        raise Stop(f"The build stopped ({(creation or {}).get('status')}). Try again, or open it in the app.")
    receipt = creation.get("authoring_receipt") or {}
    ui.p(s.green("✓ ") + f"Built in {clock(time.monotonic() - started)}"
         + (s.dim(f" · {receipt['model']}") if receipt.get("model") else ""))
    for note in creation.get("authoring_notes") or []:   # what the engine tidied to read the model's output
        ui.p(s.dim(f"  Note: {note}"))
    if creation.get("new_version_of"):
        ui.p(s.dim(f"  A new version of “{creation['new_version_of']}”"))
    ui.p()
    show_workflow(ui, creation.get("workflow"))
    ui.p()
    iid = creation["instance"]["instance_id"]
    if not (args.save or args.run or ui.ask("  Save this workflow? [Y/n] ")):
        ui.p(s.dim("  Not saved: it stays a draft."))
        return
    _, saved = client.post(f"{API}/{iid}/confirm", {"action_id": new_action("confirm"),
                                                     "draft_id": creation["draft_id"]})
    version = (saved or {}).get("version")
    ui.p(s.green("✓ ") + f"Saved as version {version}")
    if args.run:
        ui.p()
        run_workflow(ui, client, iid, version, (saved or {}).get("workflow") or creation.get("workflow"), args)
    else:
        ui.p(s.dim(f"  Run it: daoharness run {iid[:8]}"))


# -- runs -----------------------------------------------------------------------------------------------------------

def show_fields(ui: UI, content: dict, lines_each: int = 3) -> None:
    for name, value in (content or {}).items():
        rows = wrap(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), width(), "    ")
        ui.p("  " + ui.s.dim(humanize(name)))
        for row in rows[:lines_each]:
            ui.p(row)
        if len(rows) > lines_each:
            ui.p(ui.s.dim(f"    … {len(rows) - lines_each} more lines"))


def read_input(path: str, spec: dict) -> dict:
    text = Path(path).expanduser().read_text("utf-8")
    try:
        content = json.loads(text)
    except ValueError:
        return text_input(spec, text)
    if not isinstance(content, dict):
        raise Stop("The input file must hold a JSON object of input names to text.")
    return {str(k): v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in content.items()}


def text_input(spec: dict, text: str) -> dict:
    wanted = spec.get("input_requirements") or []
    names = [r["name"] for r in wanted if r.get("required")] or [r["name"] for r in wanted]
    if len(names) != 1:
        raise Stop("This workflow takes several inputs (" + ", ".join(names) + "). "
                   "Put them in a JSON file and use --input FILE.")
    return {names[0]: text}


def example_input(ui: UI, client: Client, iid: str, version, fresh: bool):
    """The newest example made for this version, or a new one (the model makes it up; no real data). None when the
    server does not make examples (the local server): the person pastes the input instead."""
    if not fresh:
        _, snap = client.get(f"{API}/{iid}")
        for sample_id in reversed(((snap or {}).get("creation") or {}).get("sample_ids") or []):
            try:
                _, sample = client.get(f"{API}/{iid}/samples/{sample_id}")
            except ApiError:
                continue
            if isinstance(sample, dict) and sample.get("version") == version:
                return sample
    try:
        status, payload = client.post(f"{API}/{iid}/samples", {"action_id": new_action("sample"), "version": version})
    except ApiError as err:
        if err.status in (404, 405):
            return None
        raise
    return settle(ui, client, status, payload, "Making up an example input")


def paste_input(ui: UI, spec: dict) -> dict:
    """Each input typed or pasted in the terminal, ended by a line with only a dot; piped text for one input."""
    s, wanted = ui.s, spec.get("input_requirements") or []
    if not wanted:
        raise Stop("This workflow lists no inputs. Give them with --input FILE.")
    if not ui.interactive:
        if len(wanted) == 1:
            return {wanted[0]["name"]: sys.stdin.read().strip()}
        raise Stop("Give the inputs with --input FILE, a JSON object with: " + ", ".join(r["name"] for r in wanted))
    ui.p(s.bold("Your input") + s.dim("  paste or type each one, then a line with only a dot (.)"))
    content = {}
    for wanted_input in wanted:
        required = bool(wanted_input.get("required"))
        ui.p("  " + s.bold(humanize(wanted_input["name"]))
             + s.dim(f"  {wanted_input.get('description') or ''}" + ("" if required else "  (optional: a dot skips it)")))
        lines: list = []
        while True:
            try:
                line = ui.prompt("  │ ")
            except EOFError:
                break
            if line.strip() == ".":
                break
            lines.append(line)
        text = "\n".join(lines).strip()
        if text or required:
            content[wanted_input["name"]] = text
    return content


def run_workflow(ui: UI, client: Client, iid: str, version, workflow, args: argparse.Namespace) -> None:
    s, spec = ui.s, (workflow or {}).get("spec") or {}
    sample = None
    if args.input:
        content = read_input(args.input, spec)
    elif args.text:
        content = text_input(spec, args.text)
    else:
        sample = example_input(ui, client, iid, version, args.new_example)
        content = paste_input(ui, spec) if sample is None else None
    if sample is None:
        body = {"version": version, "content": content}
        ui.p(s.bold("Input") + s.dim("  from you"))
        show_fields(ui, content)
    else:
        body = {"version": version, "sample_id": sample["sample_id"]}
        ui.p(s.bold("Example input") + s.dim("  made up by the model · no real data"))
        show_fields(ui, sample.get("content") or {})
    ui.p()
    action = new_action("run")
    _, view = client.post(f"{API}/{iid}/runs", {"action_id": action, "input": body})
    ui.p(s.bold("Run"))
    follow(ui, client, iid, action, view, spec)


def sources_of(record: dict) -> list:
    found = []
    for research in [record.get("research")] + [i.get("research") for i in record.get("items") or []]:
        found += [src for src in (research or {}).get("sources") or [] if src.get("status") != "dropped"]
    return found


def source_summary(sources: list) -> str:
    n, statuses = len(sources), {src.get("status") for src in sources}
    if statuses == {"verified"}:
        how = "checked on their pages" if n > 1 else "checked on its page"
    elif "grounded" in statuses:
        how = "via Google Search"
    else:
        how = "cited"
    return f"{n} source{'s' if n != 1 else ''} {how}"


def site(source: dict) -> str:
    host = urllib.parse.urlsplit(source.get("url") or "").netloc.lower()
    if not host or "vertexaisearch" in host or host.endswith("googleusercontent.com"):
        host = str(source.get("title") or host)  # grounded results link through Google; the title names the site
    return host[4:] if host.startswith("www.") else host


def chips_from(markup: str, chips: list) -> list:
    """Google's search suggestions (HTML) as (text, link) pairs: a terminal can show them as links, not as HTML."""
    for href, text in re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', markup or "", flags=re.S):
        pair = (html.unescape(re.sub(r"<[^>]+>", "", text)).strip(), html.unescape(href))
        if pair[0] and pair not in chips:
            chips.append(pair)
    return chips


def search_chips(record: dict) -> list:
    """The search suggestions of a grounded step (or, without them, its queries as Google links)."""
    chips: list = []
    for research in [record.get("research")] + [i.get("research") for i in record.get("items") or []]:
        markup = (research or {}).get("search_suggestions") or ""
        chips_from(markup, chips)
        if not markup:
            for query in (research or {}).get("queries") or []:
                pair = (query, "https://www.google.com/search?q=" + urllib.parse.quote_plus(query))
                if pair not in chips:
                    chips.append(pair)
    return chips


def show_chips(ui: UI, chips: list, indent: str) -> None:
    if chips:
        ui.p(indent + ui.s.dim("↳ Google Search: ")
             + ui.s.dim("  ".join(ui.s.link(href, f"[{clip(text, 40)}]") for text, href in chips[:3])))


def decisions_of(record: dict) -> str:
    counts: dict = {}
    entries = list(record.get("decisions") or [])
    for item in record.get("items") or []:
        entries += item.get("decisions") or []
    for entry in entries:
        if entry.get("accepted"):
            counts[entry.get("rung")] = counts.get(entry.get("rung"), 0) + 1
    order = sorted((k for k in counts if k not in RUNGS), key=str) + [k for k in RUNGS if k in counts]
    return ("decided: " + ", ".join(f"{RUNGS.get(r, 'decision engine')} {counts[r]}" for r in order)) if counts else ""


def step_seconds(record: dict):
    started, finished = parse_time(record.get("started_at")), parse_time(record.get("finished_at"))
    try:
        return max(0.0, (finished - started).total_seconds()) if started and finished else None
    except TypeError:
        return None


class RunPrinter:
    """Prints each step once as it finishes; the running step shows on the live line."""

    def __init__(self, ui: UI, spec: dict) -> None:
        self.ui = ui
        steps = (spec or {}).get("steps") or []
        self.steps = {step["id"]: step for step in steps}
        self.order = [step["id"] for step in steps]
        self.noun = item_noun(steps)
        self.shown: dict = {}
        self.first_seen: dict = {}
        self.running = None

    def number(self, step_id: str) -> str:
        return str(self.order.index(step_id) + 1) if step_id in self.order else "·"

    def role(self, record: dict) -> str:
        return (self.steps.get(record["step_id"]) or {}).get("role") or record.get("role") or "A person"

    def title(self, record: dict, limit: int) -> str:
        step = self.steps.get(record["step_id"])
        return first_line(step.get("instructions") or "", limit) if step else humanize(record["step_id"])

    def update(self, view: dict) -> None:
        self.running = None
        for record in ((view or {}).get("workflow_run") or {}).get("steps") or []:
            step_id, state = record.get("step_id"), record.get("state")
            if state == "running":
                self.first_seen.setdefault(step_id, time.monotonic())
                self.running = self.running or record
            elif self.shown.get(step_id) != state and state in ("completed", "needs_attention", "failed",
                                                                  "needs_access", "rejected"):
                self.shown[step_id] = state
                self.done(record) if state == "completed" else self.problem(record)

    def done(self, record: dict) -> None:
        s, n = self.ui.s, self.number(record["step_id"])
        if record.get("primitive") == "human_approval":
            output = record.get("output") if isinstance(record.get("output"), dict) else {}
            self.ui.p("  " + s.green(f"✓ {n:>2}  ") + s.bold(f"{self.role(record)} approved")
                      + (" with edits" if output.get("edited") else ""))
            return
        extras = []
        items = [i for i in record.get("items") or [] if i.get("state") != "excluded"]
        if items:
            extras.append(f"× {len(items)} {plural(self.noun or 'item', len(items))}")
        sources = sources_of(record)
        if sources:
            extras.append(source_summary(sources))
        decided = decisions_of(record)
        if decided:
            extras.append(decided)
        took = step_seconds(record)
        if took is not None and took >= 1:
            extras.append(f"{took:.0f} s")
        title = self.title(record, width() - 9)
        tail = " · ".join(extras)
        if tail and len(title) + len(tail) + 11 <= width():
            self.ui.p("  " + s.green(f"✓ {n:>2}  ") + title + s.dim("  " + tail))
        else:
            self.ui.p("  " + s.green(f"✓ {n:>2}  ") + title)
            if tail:
                self.ui.p("        " + s.dim(clip(tail, width() - 10)))
        if sources:
            sites = []
            for source in sources:
                if site(source) and site(source) not in sites:
                    sites.append(site(source))
            self.ui.p("        " + s.dim(clip("↳ " + " · ".join(sites), width() - 10)))
            if any(src.get("status") == "verified" for src in sources):
                quote = next(src for src in sources if src.get("status") == "verified")
                self.ui.p("        " + s.dim(clip(f"↳ “{quote.get('quote')}” ({SOURCE_LABELS['verified']})", width() - 10)))
            show_chips(self.ui, search_chips(record), "        ")

    def problem(self, record: dict) -> None:
        s, n, state = self.ui.s, self.number(record["step_id"]), record.get("state")
        if state == "rejected":
            self.ui.p("  " + s.yellow(f"✗ {n:>2}  ") + f"{self.role(record)} rejected it")
            return
        error = record.get("error") or {}
        reason = error.get("message") or error.get("detail") or error.get("code") or ""
        stuck = [i for i in record.get("items") or [] if i.get("state") in ("needs_attention", "failed")]
        if stuck and not reason:
            reason = f"{len(stuck)} of {len(record.get('items') or [])} {plural(self.noun or 'item', 2)}"
        self.ui.p("  " + s.red(f"! {n:>2}  ") + self.title(record, 48)
                  + s.dim(f"  {str(state).replace('_', ' ')}" + (f": {reason}" if reason else "")))

    def live_text(self) -> str:
        record = self.running
        if record is None:
            return "Working…"
        step_id = record["step_id"]
        progress = ""
        items = [i for i in record.get("items") or [] if i.get("state") != "excluded"]
        if items:
            finished = sum(1 for i in items if i.get("state") == "completed")
            progress = f"  {finished}/{len(items)} {plural(self.noun or 'item', len(items))}"
        elapsed = clock(time.monotonic() - self.first_seen.get(step_id, time.monotonic()))
        return f"{self.number(step_id)}  {self.title(record, 56)}{progress}  {elapsed}"


def follow(ui: UI, client: Client, iid: str, action: str, view: dict, spec: dict) -> None:
    printer = RunPrinter(ui, spec)
    while True:
        printer.update(view)
        status = (view or {}).get("status")
        if status == "waiting_approval":
            ui.live.clear()
            decided = decide_for_run(ui, client, view, f"{iid}.{action}", spec)
            if decided is not None:
                view = decided
                continue
            waiting = next((r for r in (view.get("workflow_run") or {}).get("steps") or []
                            if r.get("state") == "waiting_approval"), None)
            role = printer.role(waiting) if waiting else "a person"
            ui.live.set(f"Waiting for {role} to decide in the app")
            time.sleep(5)
        elif status in SETTLED:
            ui.live.clear()
            view = finish(ui, client, iid, action, view)
            if view is None:
                return
            continue
        else:
            ui.live.set(printer.live_text)
            time.sleep(POLL_SECONDS)
        _, view = client.get(f"{API}/{iid}/runs/{action}")


def decide_for_run(ui: UI, client: Client, view: dict, run_id: str, spec: dict):
    pending = set((view.get("workflow_run") or {}).get("pending_approval_ids") or [])
    _, listing = client.get(f"{API}/approvals")
    mine = [a for a in (listing or {}).get("approvals") or []
            if a.get("run_id") == run_id or a.get("approval_id") in pending]
    return decide(ui, client, mine[0], spec) if mine else None


def value_lines(value, limit: int, indent: str = "") -> list:
    if value is None or value == "" or value == [] or value == {}:
        return []
    if isinstance(value, bool):
        return [indent + ("yes" if value else "no")]
    if isinstance(value, (int, float, str)):
        return wrap(value, limit, indent)
    if isinstance(value, list):
        lines = []
        for entry in value:
            if isinstance(entry, dict):
                entry = "; ".join(f"{humanize(k)}: {v}" for k, v in entry.items()
                                  if isinstance(v, (str, int, float)) and v != "")
            lines += wrap(f"• {entry}", limit, indent, hang="  ")
        return lines
    if isinstance(value, dict):
        entries = value.get("items")
        if isinstance(entries, list) and entries and all(isinstance(e, dict) and "key" in e for e in entries):
            lines = []
            for entry in entries:  # one entry per item
                lines.append(f"{indent}▸ {entry.get('label') or entry.get('key')}")
                lines += value_lines(entry.get("output"), limit, indent + "  ")
            return lines
        lines = []
        first = ("title", "to", "subject", "summary", "body", "message")
        for key, entry in sorted(value.items(), key=lambda kv: first.index(kv[0]) if kv[0] in first else len(first)):
            if entry is None or entry == "" or entry == [] or entry == {}:
                continue
            if isinstance(entry, (str, int, float)) and not isinstance(entry, bool) and "\n" not in str(entry) \
                    and len(indent) + len(str(key)) + len(str(entry)) + 2 <= limit:
                lines.append(f"{indent}{humanize(key)}: {entry}")
            else:
                inner = value_lines(entry, limit, indent + "  ")
                if inner:
                    lines.append(f"{indent}{humanize(key)}:")
                    lines += inner
        return lines
    return [indent + clip(json.dumps(value, ensure_ascii=False), limit - len(indent))]


def show_content(ui: UI, approval: dict, spec: dict) -> None:
    s, steps = ui.s, {step["id"]: step for step in (spec or {}).get("steps") or []}
    lines = []
    for ref, value in (approval.get("content") or {}).items():
        kind, _, name = str(ref).partition(":")
        step = steps.get(name) if kind == "step" else None
        label = f"{step.get('role')} · {first_line(step.get('instructions') or '', 50)}" if step else humanize(name or ref)
        lines.append(s.dim(label))
        lines += value_lines(value, width() - 10)
    limit = len(lines) if ui.full else 22
    for line in lines[:limit]:
        ui.p("    " + s.dim("│ ") + line)
    if len(lines) > limit:
        ui.p("    " + s.dim(f"│ … {len(lines) - limit} more lines (add --full to see everything)"))


def decide(ui: UI, client: Client, approval: dict, spec: dict) -> dict:
    s = ui.s
    if approval.get("kind") == "decision_review":
        return review(ui, client, approval)
    role = approval.get("approver_role") or "A person"
    ui.p()
    ui.p("  " + s.yellow("⏸  ") + s.bold(f"{role} approves: ") + (approval.get("what_is_approved") or ""))
    show_content(ui, approval, spec)
    for flag in approval.get("flags") or []:
        about = f"{flag['item_label']}: " if flag.get("item_label") else ""
        ui.p("    " + s.yellow("⚑ ") + clip(about + str(flag.get("detail") or flag.get("check")), width() - 8))
    chips: list = []
    for markup in approval.get("search_suggestions") or []:
        chips_from(markup, chips)
    show_chips(ui, chips, "    ")
    body = review_batch(ui, approval) if approval.get("itemized") and approval.get("item_keys") \
        else review_whole(ui, approval)
    _, view = client.post(f"{API}/approvals/{approval['approval_id']}/decision", {
        **body, "action_id": new_action("decide"), "content_digest": approval["content_digest"]})
    if body["decision"] == "reject":
        ui.p("  " + s.yellow(f"✗ {role} rejected it: nothing goes out"))
    elif body["decision"] == "edit":
        ui.p("  " + s.green(f"✎ {role} approved it with edits"))
    elif body["decision"] == "review_items":
        verdicts = [d["decision"] for d in body["item_decisions"].values()]
        ui.p("  " + s.green(f"✓ {verdicts.count('approve')} approved")
             + (s.yellow(f" · ✗ {verdicts.count('reject')} rejected") if "reject" in verdicts else "")
             + (s.green(f" · ✎ {verdicts.count('edit')} edited") if "edit" in verdicts else ""))
    ui.p()
    return view


def review_whole(ui: UI, approval: dict) -> dict:
    if ui.yes:
        return {"decision": "approve"}
    if not ui.interactive:
        raise Stop("Waiting for your decision. Run this in a terminal, or add --yes.")
    while True:
        answer = ui.prompt("    Approve? [Y/n · e = edit first] ").strip().lower()
        if answer in ("", "y", "yes", "是", "好"):
            return {"decision": "approve"}
        if answer in ("n", "no", "否", "不"):
            return {"decision": "reject"}
        if answer in ("e", "edit"):
            edited = edit_json(ui, approval.get("content") or {})
            if edited is not None:
                return {"decision": "edit", "edited_content": edited}


def item_content(approval: dict, key: str) -> dict:
    """What a batch approval holds for one item: {ref: that item's output} for every per-item input."""
    content = {}
    for ref in approval.get("item_refs") or []:
        entries = ((approval.get("content") or {}).get(ref) or {}).get("items") or []
        content[ref] = next((entry.get("output") for entry in entries if entry.get("key") == key), None)
    return content


def review_batch(ui: UI, approval: dict) -> dict:
    """One decision for the batch: approve all, or reject and edit single items; the rest are approved."""
    s, keys = ui.s, list(approval["item_keys"])
    labels = {}
    for ref in approval.get("item_refs") or []:
        for entry in ((approval.get("content") or {}).get(ref) or {}).get("items") or []:
            labels.setdefault(entry.get("key"), entry.get("label") or entry.get("key"))
    for n, key in enumerate(keys, 1):
        ui.p(f"    {n:>2}  " + clip(labels.get(key) or key, width() - 10))
    if ui.yes:
        return {"decision": "approve"}
    if not ui.interactive:
        raise Stop("Waiting for your decision. Run this in a terminal, or add --yes.")
    verdicts: dict = {}
    while True:
        answer = ui.prompt(f"    Enter = approve {'the rest' if verdicts else f'all {len(keys)}'}"
                           " · r 2 3 = reject · e 1 = edit · n = reject all: ").strip().lower()
        if not answer:
            break
        if answer in ("n", "no"):
            return {"decision": "reject"}
        found = re.fullmatch(r"([re])\s*([\d ,]+)", answer)
        if not found:
            ui.p(s.dim("    For example: r 2 rejects item 2, e 1 edits item 1."))
            continue
        for n in sorted({int(x) for x in re.findall(r"\d+", found.group(2)) if 1 <= int(x) <= len(keys)}):
            key = keys[n - 1]
            if found.group(1) == "r":
                verdicts[key] = {"decision": "reject"}
                ui.p("    " + s.yellow(f"✗ {n} rejected"))
            else:
                edited = edit_json(ui, item_content(approval, key))
                if edited is not None:
                    verdicts[key] = {"decision": "edit", "edited_content": edited}
                    ui.p("    " + s.green(f"✎ {n} edited"))
    if not verdicts:
        return {"decision": "approve"}
    return {"decision": "review_items",
            "item_decisions": {key: verdicts.get(key, {"decision": "approve"}) for key in keys}}


def edit_json(ui: UI, value: dict):
    """The value in your editor ($VISUAL or $EDITOR) as JSON; None when unchanged or no longer the same fields."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or ("nano" if shutil.which("nano") else "vi")
    fd, path = tempfile.mkstemp(prefix="daoharness-edit-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        ui.live.restore()
        subprocess.call(shlex.split(editor) + [path])
        edited = json.loads(Path(path).read_text("utf-8"))
    except (OSError, ValueError) as exc:
        ui.p(ui.s.yellow(f"    Not changed: {exc}"))
        return None
    finally:
        os.unlink(path)
    if not isinstance(edited, dict) or set(edited) != set(value):
        ui.p(ui.s.yellow("    Not changed: keep the same fields."))
        return None
    return edited if edited != value else None


def review(ui: UI, client: Client, approval: dict) -> dict:
    """The models were not sure: a person makes the call, with what each rung answered."""
    s, items, choices = ui.s, approval.get("items") or [], {}
    ui.p()
    ui.p("  " + s.yellow("⏸  ") + s.bold("Not sure, so it asks you") + s.dim(f" · {len(items)} call{'s' if len(items) != 1 else ''}"))
    for item in items:
        options = [str(o) for o in item.get("options") or []]
        for line in wrap(item.get("question") or "Choose", width(), "    ", hang="  ")[:3]:
            ui.p(s.bold(line))
        if item.get("item_label"):
            ui.p("      " + s.dim(clip(item["item_label"], width() - 8)))
        for line in wrap(item.get("context_excerpt") or "", width() - 4, "      ")[:3]:
            ui.p(s.dim(line))
        ranked = sorted(item.get("candidates") or [], key=lambda c: -(c.get("confidence") or 0))
        for candidate in ranked:
            sure = f" ({round(candidate['confidence'] * 100)}% sure)" if isinstance(candidate.get("confidence"), (int, float)) else ""
            ui.p("      " + s.dim(f"{RUNGS.get(candidate.get('rung'), 'decision engine')} said “{candidate.get('choice')}”{sure}"))
        suggested = next((c.get("choice") for c in ranked if c.get("choice") in options), options[0] if options else None)
        for n, option in enumerate(options, 1):
            ui.p(f"      {n}. {option}" + (s.dim("   ← suggested") if option == suggested else ""))
        choices[item["id"]] = pick(ui, options, suggested)
    _, view = client.post(f"{API}/approvals/{approval['approval_id']}/decision", {
        "action_id": new_action("decide"), "decision": "choose", "content_digest": approval["content_digest"],
        "choices": choices})
    ui.p("  " + s.green("✓ Decided") + s.dim(" · the run goes on"))
    ui.p()
    return view


def pick(ui: UI, options: list, suggested) -> str:
    if ui.yes or not options:
        return suggested
    if not ui.interactive:
        raise Stop("Waiting for your decision. Run this in a terminal, or add --yes.")
    default = options.index(suggested) + 1
    while True:
        answer = ui.prompt(f"      Choose 1-{len(options)} [Enter = {default}] ").strip()
        if not answer:
            return suggested
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]


def finish(ui: UI, client: Client, iid: str, action: str, view: dict):
    """How the run ended: its result and usage, or what stopped it. A new view when the person chose to retry or
    continue, else None."""
    s, status = ui.s, view.get("status")
    run = view.get("workflow_run") or {}
    if status == "complete":
        show_result(ui, view.get("result"))
        if view.get("delivered_to"):
            ui.p(s.dim(f"  Saved to {tilde(view['delivered_to'])}"))
        show_usage(ui, view.get("usage"), run.get("metrics") or {})
        return None
    if status == "rejected":
        ui.p(s.yellow("Stopped at the approval: nothing goes out."))
        show_usage(ui, view.get("usage"), run.get("metrics") or {})
        return None
    reasons = {"needs_access": "It needs access to: " + ", ".join(view.get("missing") or run.get("missing_sources") or []),
               "interrupted": view.get("message") or "The run stopped before it finished.",
               "needs_capability": "A step needs something this workflow cannot do yet."}
    ui.p()
    ui.p(s.red("! ") + s.bold(reasons.get(status) or view.get("message")
                              or f"The run stopped: {str(status).replace('_', ' ')}" + (f" ({view['code']})" if view.get("code") else "")))
    next_action = run.get("next_action")
    if next_action in ("retry", "resume") and ui.interactive and not ui.yes \
            and ui.ask("  Try again? [Y/n] " if next_action == "retry" else "  Continue the run? [Y/n] "):
        _, again = client.post(f"{API}/{iid}/runs/{action}/{next_action}", {})
        return again
    ui.p(s.dim("  Pick it up later: daoharness watch"))
    return None


def show_result(ui: UI, result) -> None:
    if not isinstance(result, dict):
        return
    s = ui.s
    ui.p()
    ui.p(s.bold("Result  " + (result.get("title") or "")))
    rows = wrap(result.get("summary") or "", width(), "  ")
    for row in rows[:5]:
        ui.p(row)
    if len(rows) > 5:
        ui.p(s.dim(f"  … {len(rows) - 5} more lines"))
    criteria = result.get("criteria") or []
    if criteria:
        met = sum(1 for c in criteria if c.get("met"))
        mark = s.green("✓ ") if met == len(criteria) else s.yellow("! ")
        ui.p("  " + mark + f"{met} of {len(criteria)} done-checks met")


def show_usage(ui: UI, usage, metrics: dict) -> None:
    """What a run used: model calls and, when the server reports them, tokens."""
    s, usage = ui.s, usage if isinstance(usage, dict) else {}
    calls = (usage.get("calls") or 0) + (usage.get("decision_calls") or 0) or metrics.get("model_calls") or 0
    parts = [f"{calls} model call{'s' if calls != 1 else ''}"] if calls else []
    if usage.get("input_tokens") is not None:
        parts.append(f"{usage['input_tokens']:,} tokens in · {usage.get('output_tokens') or 0:,} out"
                     + (" (estimated)" if usage.get("estimated") else ""))
    if metrics.get("elapsed_seconds"):
        parts.append(f"{clock(metrics['elapsed_seconds'])} start to finish")
    if parts:
        ui.p()
        ui.p(s.bold("Usage  ") + s.dim(" · ".join(parts)))


def cmd_run(ui: UI, client: Client, args: argparse.Namespace) -> None:
    snap = resolve(client, args.workflow, saved=True)
    creation = snap.get("creation") or {}
    if creation.get("status") != "confirmed":
        raise Stop("That workflow is still a draft. Save it first: daoharness new")
    banner(ui, client)
    show_workflow(ui, creation.get("workflow"), heading="Workflow")
    ui.p()
    run_workflow(ui, client, instance_id(snap), creation.get("version"), creation.get("workflow"), args)


def cmd_approvals(ui: UI, client: Client, args: argparse.Namespace) -> None:
    s = ui.s
    _, listing = client.get(f"{API}/approvals")
    pending = (listing or {}).get("approvals") or []
    if not pending:
        ui.p(s.green("✓ ") + "Nothing is waiting for you.")
        return
    approval = pending[0]
    if len(pending) > 1:
        ui.p(s.bold(f"{len(pending)} waiting for you"))
        for n, entry in enumerate(pending, 1):
            ui.p(f"  {n}. {entry.get('workflow_domain')}"
                 + s.dim(f" · {entry.get('approver_role')} · {clip(entry.get('what_is_approved') or '', 50)}"))
        if ui.interactive and not ui.yes:
            labels = [str(n) for n in range(1, len(pending) + 1)]
            approval = pending[labels.index(pick(ui, labels, "1"))]
        ui.p()
    iid, action = approval["instance_id"], approval["run_id"].partition(".")[2]
    spec = spec_for(client, iid, approval.get("version"))
    banner(ui, client)
    ui.p(s.bold(f"Workflow  {approval.get('workflow_domain') or ''}") + s.dim(f"   version {approval.get('version')}"))
    _, view = client.get(f"{API}/{iid}/runs/{action}")
    content = (view or {}).get("input") or {}
    if content.get("content"):
        kind = "Example input" if content.get("source_kind") == "generated_sample" else "Input"
        ui.p(s.dim(f"  {kind}: " + clip(" · ".join(str(v) for v in content["content"].values()), width() - 20)))
    ui.p()
    ui.p(s.bold("Run"))
    follow(ui, client, iid, action, view, spec)


def cmd_watch(ui: UI, client: Client, args: argparse.Namespace) -> None:
    s = ui.s
    snap = resolve(client, args.workflow)
    runs = snap.get("runs") or []
    if args.run:
        runs = [r for r in runs if str(r.get("action_id", "")).endswith(args.run)]
    if not runs:
        raise Stop("No runs to follow. Start one: daoharness run")
    run = runs[0]
    iid = instance_id(snap)
    spec = spec_for(client, iid, run.get("version"))
    _, view = client.get(f"{API}/{iid}/runs/{run['action_id']}")
    ui.p(s.bold(f"Workflow  {workflow_name(snap)}") + s.dim(f"   version {run.get('version')}"))
    ui.p()
    ui.p(s.bold("Run"))
    follow(ui, client, iid, run["action_id"], view, spec)


def cmd_serve(ui: UI, args: argparse.Namespace) -> None:
    try:
        from daoharness_server.app import serve
    except ImportError:
        raise Stop('The local server needs its extras: pip install "daoharness[server]"') from None
    try:
        serve(home=HOME, port=args.port)
    except ValueError as err:   # no model configured
        raise Stop(f"{err}\nSee the README: Use your own model.") from None


# -- entry ----------------------------------------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", help="the server's address (default: hosted if you signed in, else the local server)")
    common.add_argument("--no-color", action="store_true", help="plain output")
    running = argparse.ArgumentParser(add_help=False)
    running.add_argument("--input", metavar="FILE", help="run on your own input: a JSON object of input names to text")
    running.add_argument("--text", help="run on this text (workflows with one input)")
    running.add_argument("--new-example", action="store_true", help="hosted: make up a new example input")
    running.add_argument("--yes", action="store_true", help="save, approve and choose the suggestion without asking")
    running.add_argument("--full", action="store_true", help="show everything under review")

    root = argparse.ArgumentParser(prog="daoharness", description="Build, run and approve workflows from your terminal.",
                                   epilog="Start with: daoharness serve (then, in another terminal: daoharness new)")
    root.add_argument("--version", action="version", version=f"daoharness {VERSION}")
    sub = root.add_subparsers(dest="command", metavar="COMMAND")
    serve = sub.add_parser("serve", help="start the local server with your model (127.0.0.1 only)")
    serve.add_argument("--port", type=int, default=8767)
    login = sub.add_parser("login", parents=[common], help="use DaoHarness hosted (saved only on this computer)")
    login.add_argument("--from-env", action="store_true", help="take the session from DAOHARNESS_SESSION")
    sub.add_parser("logout", parents=[common], help="forget the hosted session; back to the local server")
    new = sub.add_parser("new", parents=[common, running], help="build a workflow from one sentence")
    new.add_argument("sentence", nargs="*", help="the job, in one sentence (or type it when asked)")
    new.add_argument("--save", action="store_true", help="save without asking")
    new.add_argument("--run", action="store_true", help="save, then run it")
    run = sub.add_parser("run", parents=[common, running], help="run a saved workflow")
    run.add_argument("workflow", nargs="?", help="part of its name or id (default: the most recent)")
    approvals = sub.add_parser("approvals", aliases=["inbox"], parents=[common, running],
                               help="decide what is waiting for you")
    approvals.set_defaults(command="approvals")
    watch = sub.add_parser("watch", parents=[common, running], help="follow a workflow's latest run")
    watch.add_argument("workflow", nargs="?")
    watch.add_argument("--run", metavar="ACTION", help="a specific run (the end of its id)")
    sub.add_parser("ls", parents=[common], help="list your workflows")
    show = sub.add_parser("show", parents=[common], help="show a workflow's steps")
    show.add_argument("workflow", nargs="?")
    sub.add_parser("status", parents=[common], help="check the connection and your session")
    return root


COMMANDS = {"new": cmd_new, "run": cmd_run, "approvals": cmd_approvals, "inbox": cmd_approvals, "watch": cmd_watch, "ls": cmd_ls,
            "show": cmd_show, "status": cmd_status}


def main(argv=None) -> int:
    root = parser()
    args = root.parse_args(argv)
    if not args.command:
        root.print_help()
        return 0
    ui = UI(args)
    host = "the server"
    try:
        if args.command in ("login", "logout", "serve"):
            {"login": cmd_login, "logout": cmd_logout, "serve": cmd_serve}[args.command](ui, args)
            return 0
        client = connect(ui, args)
        host = client.host
        COMMANDS[args.command](ui, client, args)
        return 0
    except Stop as stop:
        ui.live.restore()
        print(ui.s.red("✗ ") + str(stop), file=sys.stderr)
        return 1
    except ApiError as err:
        ui.live.restore()
        print(ui.s.red("✗ ") + explain(err, host), file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        ui.live.restore()
        print("\n" + ui.s.dim("Stopped. Anything already started keeps going on the server."), file=sys.stderr)
        return 130
    finally:
        ui.live.restore()


if __name__ == "__main__":
    sys.exit(main())
