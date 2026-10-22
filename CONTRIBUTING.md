# Contributing to DaoHarness

Thanks for helping. The most useful contributions right now:

1. **Adapters.** A native `ModelPort` for a model API that does not speak the OpenAI chat completions format, or a
   `WorkflowStore` for a database. `daoharness/adapters/` has the OpenAI-compatible model; `daoharness/testing.py`
   has stand-in versions of every port to copy from. Adapters inside `daoharness/` use the standard library only;
   one that needs a provider's SDK belongs in its own package.
2. **Example workflows.** A JSON file in `daoharness_demo/examples/` with a one-sentence need, a spec and a fictional
   sample input. Run it with `daoharness-demo --spec your-file.json`.
3. **Reports from real use.** Ran DaoHarness on a real job? Open an issue with the workflow and what happened.
   Remove client data first.

## Ground rules

- The engine stays industry-neutral and depends only on the standard library and `jsonschema`.
  `tests/test_industry_neutral.py` enforces both. Industry content belongs in examples, not in `daoharness/`.
- Outbound messages stay drafts that a person approves. Changes that let a workflow send on its own will not be merged.
- Sample data must be fictional. No real people, companies or identifiers.

## Development

```bash
git clone https://github.com/DaoBrewAI/DaoHarness
cd DaoHarness
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
pytest -q
daoharness-demo --list
```

By submitting a pull request you agree that your contribution is licensed under the Apache License 2.0.
