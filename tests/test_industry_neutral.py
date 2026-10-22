"""Industry-neutral guard: the engine carries no industry vocabulary, no branches on the
spec domain, and the core imports nothing outside the stdlib, jsonschema and itself."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "daoharness"
FILES = sorted(CORE.rglob("*.py"))
BANNED = re.compile(r"(?i)\b(accounts?|account(?:ing|ant)s?|bank|clients?|hire[sd]?|tax(?:es)?|cpa|bookkeep\w*|payroll|ein|w-?2|1099|"
                    r"recruit\w*|candidates?|r[eé]sum[eé]s?|hiring|applicants?|interview\w*|"
                    r"patients?|clinic\w*|lawsuits?|attorneys?|invoices?)\b")


# Contract tokens that collide with banned words: the decision-review key "candidates", the run next_action value
# "resume" and the /runs/{action}/resume route. Only these exact quoted/route forms are allowed.
ALLOWED_KEYS = re.compile(r"""(["'])(?:candidates|resume)\1|/resume\b""")
DOMAIN_BRANCH = re.compile(r"\.domain\s*==|\[\"domain\"\]\s*==|domain\s+in\s+\(")
ALLOWED_THIRD_PARTY = {"jsonschema"}


def vocabulary_hits(files, root):
    return [f"{p.relative_to(root)}:{n}: {m.group(0)}" for p in files if p.exists()
            for n, line in enumerate(p.read_text("utf-8").splitlines(), 1)
            for m in BANNED.finditer(ALLOWED_KEYS.sub("", line))]


def domain_branch_files(files):
    return [p for p in files if p.exists() and DOMAIN_BRANCH.search(p.read_text("utf-8"))]


def foreign_imports(files, root):
    """Absolute imports that are neither stdlib, an allowed third-party package, nor the core package itself."""
    hits = []
    for p in files:
        for node in ast.walk(ast.parse(p.read_text("utf-8"), filename=str(p))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue  # relative imports stay inside the package
            for name in names:
                top = name.split(".")[0]
                if (top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY
                        or name == "daoharness" or name.startswith("daoharness.")):
                    continue
                hits.append(f"{p.relative_to(root)}:{node.lineno}: {name}")
    return hits


# --- the guard ---------------------------------------------------------------------------------------------------

def test_core_has_no_industry_vocabulary():
    hits = vocabulary_hits(FILES, ROOT)
    assert not hits, "\n".join(hits)


def test_core_has_no_domain_branches():
    assert not domain_branch_files(FILES)


def test_core_imports_only_stdlib_jsonschema_and_itself():
    core = sorted(CORE.rglob("*.py"))
    assert core, "core package not found"
    hits = foreign_imports(core, ROOT)
    assert not hits, "\n".join(hits)


# --- the guard bites ---------------------------------------------------------------------------------------------

def test_vocabulary_guard_flags_banned_words_and_spares_contract_tokens(tmp_path):
    p = tmp_path / "m.py"
    p.write_text('x = "client"\n# Resume parsing for the Candidate\nreview = item["candidates"]\n'
                 "next_action = 'resume'\npath = '/runs/{a}/resume'\n", "utf-8")
    assert vocabulary_hits([p, tmp_path / "missing.py"], tmp_path) == [
        "m.py:1: client", "m.py:2: Resume", "m.py:2: Candidate"]


def test_domain_branch_guard_flags_comparisons_on_the_domain(tmp_path):
    files = []
    for n, body in enumerate(('if spec.domain == "x": pass\n', 'if d["domain"] == "x": pass\n',
                              'if domain in ("a", "b"): pass\n', 'label = spec.domain\n')):
        files.append(tmp_path / f"m{n}.py")
        files[-1].write_text(body, "utf-8")
    assert domain_branch_files(files) == files[:3]


def test_import_guard_flags_web_framework_and_app_modules(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("from __future__ import annotations\nimport json, jsonschema\nfrom . import spec\n"
                 "from daoharness.spec import REF\nimport fastapi\n"
                 "def f():\n    from myapp.web import app\n    from starlette.requests import Request\n",
                 "utf-8")
    assert foreign_imports([p], tmp_path) == [
        "m.py:5: fastapi", "m.py:7: myapp.web", "m.py:8: starlette.requests"]
