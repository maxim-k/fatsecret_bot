# Commit scopes

Commits follow [Conventional Commits](https://www.conventionalcommits.org/), **one line, no body**:

```
feat(vision): extract product and weight from scale photos
fix(match): fall back to search when alias lookup misses
chore(build): pin python 3.14 and add ruff config
```

One commit per completed task, where a task is something that leaves the repo working — not a file save.

## What a scope is

A scope is a **subsystem**: a reason the code changes. It is not a filename. `store.py` is a file; `store` is a scope only because persistence changes for its own reasons, independently of how food is recognised or submitted.

When picking a scope, ask *"what would make me touch this again?"* rather than *"which file did I edit?"*. A change spanning two scopes is usually two commits.

## Scopes

| Scope | Covers |
|---|---|
| `ingest` | Telegram intake — receive a photo, acknowledge it, deduplicate, hand off for recognition |
| `vision` | Image → product identity and weight, including prompt and response validation |
| `match` | Product text → FatSecret food, serving, and unit count; alias learning |
| `diary` | FatSecret writes and the OAuth token lifecycle |
| `review` | Pending queue and the approve / edit / undo interaction |
| `store` | Shared data models, and persistence of aliases, queue, and the audit log |
| `build` | uv, ruff, mypy, pytest, Docker, CI |

Seven scopes is generous for a project this size. If one never gets used, delete the row rather than keeping it for symmetry.

There is deliberately no `docs` scope — it would only ever restate the type. Documentation about a subsystem takes that subsystem's scope (`docs(match): explain serving ratio maths`); repo-wide documentation omits the scope entirely (`docs: define commit scopes`).

## Types

`feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`.

Use `chore` for tooling and dependency work, `build` being its usual scope. Prefer `fix` only for defects in shipped behaviour — correcting something never released is part of the `feat`.
