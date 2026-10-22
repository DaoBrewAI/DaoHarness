# AGENTS.md

Notes for coding agents, and for people, working in this repository.

## Layout

| Path | What it holds |
|---|---|
| `daoharness/` | The engine: spec, validator, compiler, executor, decisions, step types, file store, service. It uses only the standard library and `jsonschema`. |
| `daoharness/adapters/` | Model adapters (`OpenAICompatibleModel`) |
| `daoharness_server/` | The local server behind `daoharness serve` (FastAPI, 127.0.0.1 only) |
| `daoharness_cli/` | The `daoharness` terminal, for the local server and for hosted |
| `daoharness_demo/` | `daoharness-demo` and the three example workflows |
| `tests/` | pytest, with stand-in models and no network beyond 127.0.0.1 |

## Before you open a pull request

```bash
pip install -e ".[test]"
pytest -q
daoharness-demo --yes --no-color
```

## Rules

- **Keep the engine industry-neutral.** `tests/test_industry_neutral.py` fails on:
  - industry words in `daoharness/`;
  - any branch on a workflow's domain.

  Industry content belongs in examples.
- **Keep model schemas strict-mode safe.** Every schema sent to a model must pass `tests/test_model_schemas.py`.
- **Send nothing.** Messages are drafts that a person approves. Do not add code that sends them.
- **Keep secrets out.** Never commit keys, session cookies or client data.
- **Limit core imports.** The core imports only the standard library, `jsonschema` and itself.
