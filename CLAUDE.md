# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

A single-user Telegram bot. The user photographs food on kitchen scales, forwards the photo to the bot, and the bot logs it to their **FatSecret** diary. The **Physical Life** fitness app (`pht.space`) syncs that diary, so FatSecret is the write target and Physical Life is never contacted.

The problem being solved is *backlog*, not typing. Entering food by hand is slow, so it gets deferred, and deferred logging becomes no logging. Therefore: **a food the user has eaten before must require zero interaction.** Any change that adds a confirmation step to a repeat food defeats the point.

## Do not reverse-engineer the Physical Life API

This was tried and abandoned. Do not restart it without being asked.

- **HTTP proxy mode** (mitmproxy): 172 flows captured and decrypted, but zero requests to any Fizikl/PHT host — only the app's native SDKs (Amplitude, Sentry, Firebase). Flutter's `dart:io` ignores the iOS system HTTP proxy.
- **WireGuard mode**: tunnel active, keys verified, listener confirmed, firewall off, Mac↔phone ping 9 ms — `tcpdump 'udp port 51820'` captured **0 packets** across three attempts. iOS never put the handshake on the wire.

FatSecret has a documented public API and Physical Life syncs from it. That is the route.

## Architecture

```
Telegram photo ──▶ ingest (ack <1s, never blocks)
                      │
                      ▼
                 vision  ──▶ Recognition{items: [RecognizedItem]}
                      │
                      ▼
                  match  ──▶ alias hit?  ──yes+confident──▶ diary ──▶ FatSecret
                      │                                        │
                      └── no ──▶ queue ──▶ review (buttons) ────┘
```

| File | Scope | Responsibility | Must not |
|---|---|---|---|
| `models.py` | `store` | **Shared contract.** All inter-module types. | — |
| `store.py` | `store` | JSON persistence, atomic writes, audit log | Contain business logic or normalisation |
| `match.py` | `match` | Alias normalisation, gram→serving maths | Do any I/O — it is pure by design |
| `vision.py` | `vision` | Photo → validated `Recognition` | Downsample images |
| `fatsecret.py` | `diary` | OAuth1 signing, food lookup, diary read/write | Grow a retry or rate-limit framework |
| `bot.py` | `ingest`/`review` | Telegram handlers, orchestration, entry point | Hold logic that belongs in `match` |

### `models.py` is a fixed contract

Modules are written and changed independently. **Do not change `models.py` as a side effect of working on one module** — if it fights you, say so and change it deliberately, because every other module depends on the shape.

## Style rules

- **Python 3.14**, `uv` for everything. `from __future__ import annotations` at the top of every module.
- **Full type annotations.** `mypy --strict` must pass with zero findings.
- **reST docstrings** on public functions: `:param:`, `:returns:`, `:raises:`.
- **Comments explain WHY, not what.** No `# open the file`. Comment the non-obvious: why epoch days, why ml isn't grams, why the async client.
- **Pydantic at every trust boundary** — LLM output, FatSecret responses, stored JSON. Never `json.loads` an untrusted payload into a dict and index it.
- **KISS and YAGNI are explicit user requirements**, repeatedly stated. No classes where a function will do. No ABCs, no plugin registries, no config objects, no caching layers, no database.

### Deliberately not built — do not "helpfully" add these

Daily auth health-checks, scheduled nudges, digest messages, prompt caching, a `Sink` abstraction, idempotency tables, retry frameworks, SQLite. Each was considered and cut.

**Guard the model and the API. Never correct the user.** This distinction is the whole rule, and getting it backwards has annoyed the user twice.

*User behaviour is never questioned.* Removed at explicit instruction, do not reintroduce: duplicate-photo detection (send the same photo ten times, it logs ten times), "send as a file" tips, "/pending to review" prompts, and every other advisory reply.

*Model and API output is absolutely questioned*, via `bot._hold_reason`. An item is held and the user asked when: the food has no alias, the model's confidence is below `AUTO_SUBMIT_CONFIDENCE`, `vision._digits_agree` finds `raw_digits` contradicting `weight_g`, or `match.is_outlier` finds the weight unlike anything logged for that food before. That last one exists because a scale's status indicator gets read as a trailing zero — 56 g of bread became 560 g.

Errors are reported. Anomalies are asked about. The user's own input is not policed.

**One re-read before asking.** When a weight falls outside `match.plausible_range`, `bot._reread_hint` builds an instruction naming the first reading and the expected band, and `vision.recognize(..., hint=...)` looks at the same photo again. Landing back in band logs with a `(re-read; first pass said 560 g)` note; otherwise the item is held showing both readings. **Exactly one retry** — guarded by the fact that `_reread` is called once and never recurses. Both weights go to `log.jsonl` as a `reread` event.

**Corrections are replies, not conversations.** `bot.on_reply` recovers the item from the eight-character id in the quoted message: a numeric reply sets the weight, anything else names the right food and re-teaches the alias. There is deliberately no conversation state machine — `fs_bot` had two `MessageHandler`s in one group and the second never fired.

## Verification loop

Run all four before claiming anything works:

```bash
uv run ruff check .
uv run ruff format .
uv run mypy *.py
uv run pytest -q
```

## Testing policy

Mock **only** the two external boundaries, and nothing else:

- **Anthropic** — monkeypatch `vision._get_client` to return a fake. Never hit the real API.
- **FatSecret** — `respx` to intercept httpx. Never hit the real API: the key is IP-whitelisted and rate-limited.

Worth testing: the gram→serving maths, alias resolution, LLM-response validation rejecting malformed input, duplicate suppression, OAuth1 signature base strings, FatSecret's object-vs-list quirk. Not worth testing: Telegram handler plumbing. Keep the suite proportionate to ~1500 lines of source.

## Gotchas that have already caused bugs

**FatSecret**
- **OAuth 1.0 only.** Diary access requires the *Consumer Key + Shared Secret* and three-legged delegation. The OAuth 2.0 *Client ID + Client Secret* pair supports only the `client_credentials` grant, which is app-level food lookup and **cannot touch a user's diary**. Confusing the two produces a bare `400 Invalid signature`, not a helpful message — verified against the live API, and our signing was confirmed byte-identical to `oauthlib`'s reference implementation, so suspect the credentials before the code.
- `date` is **days since 1970-01-01**, not a timestamp. Use `models.to_epoch_days`.
- **`number_of_units` means different things per serving.** For a natural-unit serving ("1 slice", 39 g) it is a multiple. For a plain metric serving ("100 g") one unit is a single **gram** — 2.89 units of Mango's "100 g" serving logs 2.89 g, not 289 g. `match.grams_per_unit` encodes this; `Alias.serving_grams` stores grams-per-unit, not the serving's size.
- Telegram stamps `message.date` in **UTC**. An evening photo files against tomorrow's diary unless converted with `.astimezone()`.
- **Errors arrive as HTTP 200** with an `error` object in the body. Status codes alone are not enough.
- A single result comes back as an **object**; multiple come back as a **list**. Numbers arrive as **strings**. `fatsecret._as_list` and `_num` normalise both — use them.
- **IP whitelisting is mandatory.** Basic tier allows 15 individual addresses, no CIDR. A changing home IP or a new VPS will break auth with a confusing error.
- **All five calls verified against the live API (2026-08-15)**, including a create/read-back/delete round trip that left the diary as it started.
- **Short-term throttling is real and undocumented**: bursts return HTTP 200 with `error 12`, "User is performing too many actions". This is separate from the 5,000/day cap and clears on its own. Keep bulk reads sequential with a ~1.5 s delay; do not fan out with `asyncio.gather`.
- A food logged from a non-US regional dataset can return `error 106 Invalid ID` when looked up on the Basic tier. Skip and continue rather than aborting a bulk operation.
- Liquids (milk, protein shakes, beer) usually expose only **ml** servings, so `pick_serving` rejects them by design and they land in the review queue. That is correct, not a bug — grams and millilitres are not interchangeable.
- Basic tier is **US-only dataset**. This is a non-issue: the user lives in the US and logs US-catalogue foods exclusively. Physical Life treats the diary as a plain nutrient tracker and does not care about the food's source. Aliases are seeded from the user's own diary history rather than search (37 seeded from August 2025).

**Vision**
- `weight_g` is required and `> 0`, so the model cannot express "unreadable". **`confidence: 0.0` is the agreed signal** that the weight was not actually read; `bot.UNREADABLE` is the only confidence value that holds an item. Anything between 0 and 1 is self-graded noise — a 0.75 once held two correct readings.
- `_digits_agree` is a **containment** check, not equality. The scale prints unit and mode glyphs, so a display reading 379 arrives as `raw_digits="8379h"`. Requiring equality rejected correct readings.
- **Never downsample the image.** The scale's LCD is often a small part of the frame and resolution is what makes it legible.
- Must use `AsyncAnthropic`. A sync client inside a Telegram handler blocks the whole event loop.
- Default model is `claude-sonnet-5`. Haiku 4.5 was tried first and read a display showing **117 g as 170, 170 and 110** across three attempts; prompt work did not move it. Every photo is now kept in `DATA_DIR/photos/`, so re-run the comparison over real images before changing this again.

**Matching**
- A serving in **ml is not interchangeable with grams**. Treating 1 ml as 1 g misreports calories for anything that isn't water. `pick_serving` skips them.
- `lookup` is **not** exact string matching. The model names the same food differently between runs ("Bread Alone San Francisco Sourdough" then "bread"), so it falls back to whole-token containment and returns `None` when two aliases match rather than guessing.
- `plausible_range` widens its margin below five samples (x0.5/x2.0 instead of x0.75/x1.25). A food logged twice has not shown its real range.
- Weights only enter `Alias.weights` when the item logged **without** a hold or the user approved it. A misread that slipped through must not widen the band meant to catch misreads.

**Telegram**
- Compressed photos smear the digits and strip EXIF. **Prefer `Document` over `PhotoSize`** when both are present.

**Tooling**
- `[tool.uv] package = false` — this is an application, not a library. Tests import root modules via `pythonpath = ["."]` in the pytest config, not via an installed package.
- In the Dockerfile, the shared `deps` layer must install with `--no-dev`. If dev tooling lands in a shared lower layer, prod inherits it and merely whites it out, making the prod image *larger* than dev.

## Commits

One-line Conventional Commits, one commit per completed task. Scopes are defined in `SCOPE.md` — read it before committing. There is deliberately no `docs` scope.
