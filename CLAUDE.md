# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

A single-user Telegram bot. The user photographs food on kitchen scales, forwards the photo to the bot, and the bot logs it to their **FatSecret** diary. FatSecret is the only external service the bot writes to.

The problem being solved is *backlog*, not typing. Entering food by hand is slow, so it gets deferred, and deferred logging becomes no logging. Therefore: **a food the user has eaten before must require zero interaction.** Any change that adds a confirmation step to a repeat food defeats the point.

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

*Model and API output is absolutely questioned*, via `bot._hold_reason`. An item is held and the user asked when the food has no alias, the reading came back `UNREADABLE`, or the weight falls outside `match.plausible_range`. That last one exists because a scale's status indicator gets read as a trailing zero — 56 g of bread became 560 g.

Errors are reported. Anomalies are asked about. The user's own input is not policed.

**There is no automatic re-read. Removed deliberately, do not reintroduce.** A doubtful reading used to be sent back to a *stronger* model with a hint, which is the only thing that made it worth doing: of the three `reread` events ever logged, one changed nothing and two were Haiku's misreads corrected by Sonnet. With one model the retry asks the same reader the same question, and if the second answer happens to land in band it silently overwrote the first and logged it — picking between two equally-trustworthy readings by which came last. A doubtful reading is now held and the user asked, which is what `_hold_reason` was always for. Reviving this needs a second model that is actually better, not just another call.

**A caption is a correction supplied up front.** `bot.parse_correction` splits a caption or reply into an optional name and an optional trailing weight ("Avocado", "56", "Avocado 56"), and `_apply_correction` applies it before any hold decision. Captions and replies share that one parser deliberately — two copies would drift. Telegram attaches an album caption to the single message it was typed on, not the batch.

**`/backlog` files a batch against a past day.** `bot.parse_backlog` reads `/backlog DD/MM/YYYY [meal]` (also `DD/MM`, `today`, `yesterday`) off a caption and runs *before* `_apply_correction` — otherwise the whole directive becomes the product name and the bot searches FatSecret for `"/backlog 15/08/2026 lunch"`. **Day before month**, because accepting both orders makes `08/09` ambiguous and a silently wrong day is the exact corruption everything else here guards against; an unreadable date is refused and the batch dropped, never defaulted to today. Meal is checked against `match.MEALS` — FatSecret accepts those four and reports anything else opaquely. Omitting it gives `other`, not `meal_for(now)`, which would describe when the photo was *sent*.

**An album is collected before it is processed.** Telegram sends one update per photo and puts the caption on one of them, so a directive naming a date reaches one frame out of ten. `on_photo` stashes album frames in `_albums` and reschedules `_flush_album`, which processes the group `_ALBUM_SETTLE` seconds after the updates stop. **Only the `file_id` is stashed** — downloading in the handler would let one slow full-resolution document outlast the window and flush a half-finished batch, and the stragglers would then form a second group with no directive and land on today. A lone photo skips all of this and is processed inline.

**One `asyncio.Lock` guards the queue.** The load-modify-save of `queue.json` used to be safe purely because `max_concurrent_updates == 1`. An album flush runs as its own task *outside* that serialisation, so `bot._queue_lock` now wraps the queue span in `_process_photo`, `on_reply` and `on_button`. Do not remove it while album batching exists, and do not add a fourth queue writer without taking it.

**`/new` reads a nutrition label into a new FatSecret food.** `bot.parse_new` splits `/new Name, Brand` on the **last** comma and deliberately does not reuse `parse_correction`, which would read the trailing number in "Fiber One 10 g Fiber" as a weight and leave the name as "Fiber One". Every frame of the batch goes to `vision.read_label` in one call — the name is on the pack and the numbers are on the panel.

**Percentages are resolved from a table, not by the model.** A label may print "10% calcium" and no milligrams. `vision.read_label` returns one `NutrientReading` per line, carrying `amount`, `percent_daily_value`, or both, and `match.label_nutrients` converts against `match.DAILY_VALUES` (FDA, adults and children 4+). A printed amount always wins, because label percentages are rounded to whole numbers. Total sugar is deliberately absent from the table: the 50 g DV is for *added* sugars while FatSecret's `sugar` field is total, and labels print total sugars in grams anyway. `calories` is in `label_nutrients` despite having no DV, because `food.create.v2` requires it — leaving it out failed the live call while every mock passed.

**The response schema is part of the API contract, and mocks do not test it.** `NutritionLabel` began as a flat panel with 17 nullable fields; Anthropic caps a schema at 16 union-typed parameters, so **every `/new` failed with a 400 before the photo was ever sent** — through a full release, because every test stubbed the client and no test ever built a request. It is now a `list[NutrientReading]`, which also drops the open-ended `dict[str, float]` that Gemini's schema subset does not reliably accept. `tests/test_bot.py` guards both limits offline. Adding nullable fields to a response model is an API change, not a modelling preference.

**A new food is chosen, never guessed.** `_offer_foods` searches, shows the top three with the macros `foods.search` already returns, and marks any candidate with no gram serving before it can be tapped. The `pick:` callback resolves the serving and saves the alias. The previous `_teach` silently committed the first weighable hit as a permanent alias, so a generic food could become a branded one forever without the user seeing it.

**A logged item can still be corrected.** Replying to "Logged Egg — 108 g" with `118` deletes that FatSecret entry, writes a new one, and **replaces** the wrong weight in `Alias.weights` rather than appending — otherwise a misread keeps widening the band that exists to catch misreads. `QueuedItem.food_entry_id` exists for exactly this. Each logged item gets its own message so a reply resolves to one item.

**Callback data is split once, up front.** `action, item_id, *extra = query.data.split(":")`. Splitting on the first colon only made `pick:ID:FOOD` look up an item id of `"ID:FOOD"`, so every pick answered "That item is gone" — the picker was dead on arrival for a full release. Test the routing, not only the handlers it dispatches to.

**Corrections are replies, not conversations.** `bot.on_reply` recovers the item from the quoted message id: a numeric reply sets the weight, anything else names the right food and re-teaches the alias. There is deliberately no conversation state machine — `fs_bot` had two `MessageHandler`s in one group and the second never fired.

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

- **Gemini** — monkeypatch `vision._get_client` to return a fake. Never hit the real API. `benchmark.py` is the deliberate exception and is not a test.
- **FatSecret** — `respx` to intercept httpx. Never hit the real API: the key is IP-whitelisted and rate-limited.

Worth testing: the gram→serving maths, alias resolution, LLM-response validation rejecting malformed input, duplicate suppression, OAuth1 signature base strings, FatSecret's object-vs-list quirk. Not worth testing: Telegram handler plumbing. Keep the suite proportionate to ~1500 lines of source.

## Gotchas that have already caused bugs

**FatSecret**
- **Creating foods needs Premier; the account has Premier Free since 2026-08-16.** On Basic, `food.create` and `food.create.v2` return **error 10, "Unknown method"** — the method is absent, not merely forbidden. That is distinguishable from a malformed call, which returns 101: probe an unfamiliar method with its required parameters omitted and read the code. After the upgrade both versions answer `101 Missing required parameter: brand_name`, and `/new` works. `brand_name` is required in practice despite the docs calling it brand-only.
- **A created food cannot be deleted or edited.** There is no `food.delete` and no `food.edit` — only `food.add_favorite` / `food.delete_favorite`, which manage a list, not the food. So `food.create` is the one irreversible write in this codebase, and `bot._offer_label` shows the whole panel behind an Add/Discard tap before calling it. That confirmation is guarding the *model*, not the user: it has just transcribed twenty numbers off small print into a permanent record.
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
- Basic tier is **US-only dataset**. This is a non-issue: the user lives in the US and logs US-catalogue foods exclusively. Aliases are seeded from the user's own diary history rather than search (37 seeded from August 2025).

**Vision**
- `weight_g` is required and `> 0`, so the model cannot express "unreadable". **`confidence: 0.0` is the agreed signal** that the weight was not actually read; `bot.UNREADABLE` is the only confidence value that holds an item. Anything between 0 and 1 is self-graded noise — a 0.75 once held two correct readings.
- `_digits_agree` is a **containment** check, not equality. The scale prints unit and mode glyphs, so a display reading 379 arrives as `raw_digits="8379h"`. Requiring equality rejected correct readings.
- **Never downsample the image.** The scale's LCD is often a small part of the frame and resolution is what makes it legible.
- Must go through `client.aio`. A sync client inside a Telegram handler blocks the whole event loop.
- **One model, chosen by measurement.** `gemini-3.5-flash` with **thinking disabled** reads every photo. Against the 17-photo corpus it got 9 of 9 known weights right at a 4.7 s median, including every trailing-zero case that beat Haiku. `gemini-3.5-flash-lite` managed 4 of 11 and reproduced exactly those misreads (640 for 64 g, 1080 for 108 g); 3.6 and 3.7-flash answered 503 on every attempt. `benchmark.py` replays `DATA_DIR/photos/` against candidates — run it before changing `GEMINI_MODEL`, do not argue about it.
- **Thinking off is not a cost decision.** Automatic thinking tripled the median to 13 s and read nothing better, and thinking tokens count against `_MAX_OUTPUT_TOKENS`, so one photo returned `finish_reason=MAX_TOKENS` with no answer at all. `gemini-3.5-flash-lite` rejects `thinking_budget=0` with a bare 400.
- **The provider's own errors are wrapped.** `genai.errors.APIError` becomes `RecognitionError`, because transient 503s and free-tier 429s are routine and every caller already handles that one type.
- **A weight of `0` means "no reading".** It is a legal value, not a validation failure. `weight_g` was `gt=0`, so a nutrition-label photo sent as food returned 0, failed pydantic validation inside `recognize`, and destroyed the item — no queue entry, nothing to reply to. `bot._hold_reason` checks the zero **before** the unknown-food check, or a scale-less photo opens the picker offering to log 0 g.
- **Confidence gates nothing but the hold.** Only an explicit `0.0` holds, alongside a `weight_g` of 0. A self-graded 0.75 once held two correct readings, so nothing between 0 and 1 is worth branching on.

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
