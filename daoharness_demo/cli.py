"""daoharness-demo: build and run an example workflow on the real engine with a stand-in model.

Nothing here calls a model provider or the network. The stand-in model returns the example's workflow when asked to
build it and placeholder output for each step, so what you see is the engine itself: validation, per-item steps,
decisions, a person's approval and the delivered package. To run your own job on your own model, use the terminal:
daoharness serve, then daoharness new.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from importlib import resources
from pathlib import Path

from daoharness.testing import FakeModel, as_model_payload, service_for

ORG = "demo"
DEFAULT_EXAMPLE = "security-questionnaire"
DECISION_PRIMITIVES = {"checklist_match", "screen_score", "classify_route"}


class Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def blue(self, text: str) -> str:
        return self._wrap("38;5;63", text)

    def yellow(self, text: str) -> str:
        return self._wrap("38;5;178", text)

    def green(self, text: str) -> str:
        return self._wrap("38;5;35", text)


def load_examples() -> dict[str, dict]:
    folder = resources.files("daoharness_demo") / "examples"
    examples = {}
    for entry in folder.iterdir():
        if entry.name.endswith(".json"):
            doc = json.loads(entry.read_text("utf-8"))
            examples[doc["id"]] = doc
    return examples


def step_tag(step: dict, noun: str | None) -> str:
    """The label a person cares about most: who acts, and how often."""
    primitive = step["primitive"]
    if step.get("approval") or primitive == "human_approval":
        tag = "you approve"
    elif primitive in DECISION_PRIMITIVES:
        tag = "asks if unsure"
    elif primitive == "draft_message":
        tag = "draft only"
    else:
        tag = "automatic"
    if step.get("for_each"):
        tag += f" · per {noun or 'item'}"
    return tag


def first_sentence(text: str, limit: int = 62) -> str:
    text = text.strip().split("\n")[0]
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def wrap(text: str, width: int = 92, indent: str = "  ") -> str:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + len(word) + 1 > width - len(indent):
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return "\n".join(indent + item for item in lines)


def describe(record: dict, step: dict, noun: str | None) -> str:
    title = first_sentence(step["instructions"], 56)
    if record["primitive"] == "deliver_package":
        return f"{title} {step['config'].get('title', 'package')}".replace(". ", ": ", 1)
    items = record.get("items") or []
    if items:
        return f"{title}  × {len(items)} {noun or 'item'}s"
    return title


def print_workflow(p, style: Style, workflow: dict) -> tuple[list, str | None]:
    """The workflow, one line per step, tagged with who acts and how often."""
    metrics, steps = workflow["metrics"], workflow["spec"]["steps"]
    noun = next((s["config"].get("item_noun") for s in steps if s["primitive"] == "split_items"), None)
    summary = f"{metrics['total_steps']} steps · {round(metrics['automation_ratio'] * 100)}% automatic"
    if metrics.get("fan_out"):
        summary += f" · runs once per {noun or 'item'}"
    p(style.bold("Workflow ") + workflow["spec"]["domain"] + style.dim("  " + summary))
    role_width = max(len(s["role"]) for s in steps)
    for n, step in enumerate(steps, 1):
        tag = step_tag(step, noun)
        colour = style.yellow if tag.startswith("you approve") else style.blue
        p(f"  {n}  {step['role']:<{role_width}}  {first_sentence(step['instructions']):<62}  {colour(tag)}")
    p()
    return steps, noun


def print_usage(p, style: Style, usage: dict) -> None:
    calls = usage.get("calls") or 0
    tokens = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
    line = f"  {calls} model call{'s' if calls != 1 else ''}"
    if tokens:
        line += f" · {tokens:,} tokens" + (" (estimated)" if usage.get("estimated") else "")
    p(style.bold("Usage"))
    p(line)
    p()


def run_example(doc: dict, *, approve: bool | None, style: Style, out=sys.stdout) -> int:
    def p(text: str = "") -> None:
        print(text, file=out)

    model = FakeModel(compile_outputs=[as_model_payload(copy.deepcopy(doc["spec"]))])
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="daoharness-demo-") as tmp:
        svc = service_for(Path(tmp), model, decisions=False)
        p(style.bold("DaoHarness") + style.dim(" · open engine demo · stand-in model, no API key"))
        p()
        p(style.bold("You type"))
        p(wrap(f"“{doc['need']}”"))
        p()

        created = svc.create(ORG, "build-1", doc["need"])
        workflow = created["workflow"]
        steps, noun = print_workflow(p, style, workflow)

        iid = created["instance"]["instance_id"]
        svc.confirm(ORG, iid, "confirm-1", created["draft_id"])
        run = svc.run(ORG, iid, "run-1", {"version": 1, "content": doc["input"]})
        by_id = {s["id"]: s for s in steps}
        printed: set[str] = set()

        def show_completed(current: dict) -> None:
            for record in current["workflow_run"]["steps"]:
                if record["state"] == "completed" and record["step_id"] not in printed \
                        and record["primitive"] != "human_approval":
                    printed.add(record["step_id"])
                    number = list(by_id).index(record["step_id"]) + 1
                    p("  " + style.green(f"✓ {number} ") + describe(record, by_id[record["step_id"]], noun))

        p(style.bold("Run"))
        show_completed(run)
        decision = "approve"
        while run["status"] == "waiting_approval":
            pending = svc.approvals(ORG, ORG)["approvals"][0]
            p("  " + style.yellow(f"⏸ Waiting for {pending['approver_role']} to approve: {pending['what_is_approved']}"))
            if approve is None:
                answer = input("    Approve? [Y/n] ").strip().lower()
                decision = "reject" if answer in ("n", "no") else "approve"
            else:
                decision = "approve" if approve else "reject"
            run = svc.decide(ORG, ORG, pending["approval_id"], f"decide-{pending['approval_id']}", decision,
                             pending["content_digest"])
            p("  " + (style.green(f"✓ {pending['approver_role']} approved") if decision == "approve"
                      else style.yellow(f"✗ {pending['approver_role']} rejected: nothing goes out")))
            show_completed(run)
        p()

        if run.get("status") == "complete":
            print_usage(p, style, run.get("usage") or {})
        p(style.dim(f"Ran in {time.monotonic() - started:.1f} s on a stand-in model. To run your own job on your "
                    "own model: daoharness serve, then daoharness new."))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="daoharness-demo",
                                     description="Build and run an example workflow on the DaoHarness engine, "
                                                 "with a stand-in model.")
    parser.add_argument("example", nargs="?", default=None, help="example id (see --list)")
    parser.add_argument("--list", action="store_true", help="list the bundled examples")
    parser.add_argument("--spec", type=Path, help="run your own example file (same shape as the bundled ones)")
    parser.add_argument("--yes", action="store_true", help="approve without asking")
    parser.add_argument("--reject", action="store_true", help="reject at the approval step")
    parser.add_argument("--no-color", action="store_true", help="plain output")
    args = parser.parse_args(argv)

    approve = False if args.reject else True if (args.yes or not sys.stdin.isatty()) else None
    style = Style(enabled=sys.stdout.isatty() and not args.no_color)
    examples = load_examples()
    if args.list:
        for doc in sorted(examples.values(), key=lambda d: d["id"]):
            print(f"{doc['id']:<24} {doc['title']}")
        return 0
    if args.spec:
        doc = json.loads(args.spec.read_text("utf-8"))
    elif (args.example or DEFAULT_EXAMPLE) in examples:
        doc = examples[args.example or DEFAULT_EXAMPLE]
    else:
        parser.error(f"unknown example {args.example!r}; try --list")
    return run_example(doc, approve=approve, style=style)


if __name__ == "__main__":
    raise SystemExit(main())
