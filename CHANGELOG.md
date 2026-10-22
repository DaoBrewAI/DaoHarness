# Changelog

## Unreleased

- A new README: DaoHarness Skeleton, a recorded run, how it works, Skeleton and hosted, FAQ.
- Issue forms, a pull request template and AGENTS.md.
- Model replies with `<think>` blocks, code fences, trailing commas or capitalised step ids are tidied and recorded.
- Decisions: a choice in another case or a confidence like `"92%"` is read; an unreadable answer goes to a person alone.
- Errors say what went wrong and what to do, not "model unavailable".
- A local build shows a waiting line with the elapsed seconds and is waited for up to 15 minutes, not 60 seconds.
- Windows: the file store locks with `msvcrt` where there is no `fcntl`.
- Gemini and Vertex AI work: their endpoints start in JSON mode, any 400 or a stalled strict request steps down a mode, and the schema is in the prompt too.

## 0.1.0 · 2026-10-22

The first public release:
- the workflow engine;
- the `daoharness` terminal;
- a local server on your own model;
- an OpenAI-compatible adapter;
- three example workflows.
