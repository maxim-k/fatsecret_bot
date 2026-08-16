# pht_food_bot

Photograph your food on kitchen scales, forward it to a Telegram chat, forget about it. The bot reads the product and the weight off the scale's display, logs it to your FatSecret diary, and the Physical Life app syncs it from there.

## Why

Logging food by hand is slow — search, type, repeat for every item. Slow means deferred, deferred becomes a backlog, and a backlog becomes nothing logged at all.

So the design goal isn't "make typing faster", it's **a food you've eaten before takes zero interaction**. The first time you eat something you teach the bot what it is; every photo of it after that logs itself.

## How it works

```
photo → Claude reads product + weight → known food? → FatSecret diary → Physical Life
                                             │
                                          new food → held for one tap
```

Physical Life has no public API, so the bot never talks to it. It writes to FatSecret, and Physical Life's own FatSecret integration pulls it in.

## Setup

### 1. FatSecret

1. Register at [platform.fatsecret.com/register](https://platform.fatsecret.com/register) — free **Basic** edition, self sign-up, no approval.
2. Under **Manage API Keys**, copy the **OAuth 1.0 Consumer Key and Shared Secret**.

   > Not the OAuth 2.0 **Client ID / Client Secret**. FatSecret issues both, and only OAuth 1.0 can reach a user's diary — OAuth 2.0 supports just the `client_credentials` grant, which is app-level food lookups only. Using the wrong pair fails with a bare `400 Invalid signature`.

3. Under **IP Restrictions**, whitelist the machine that will run the bot. Basic allows 15 individual addresses and no CIDR ranges, so add your home IP for development and the VPS IP when you deploy. Auth fails confusingly if you skip this.
4. Get an access token for your own account:

   ```bash
   uv run --env-file .env python fatsecret.py authorize
   ```

   It prints a URL, you approve in a browser, FatSecret shows a PIN, you paste it back, and it prints the token and secret to put in `.env`. No callback server needed.

5. In the Physical Life app, turn the **FatSecret integration** switch on.

### 2. Telegram

Message [@BotFather](https://t.me/BotFather), send `/newbot`, and keep the token. Get your own numeric user id from [@userinfobot](https://t.me/userinfobot) — the bot ignores everyone else.

### 3. Anthropic

Create an API key at [console.anthropic.com](https://console.anthropic.com). This bills separately from any Claude subscription.

### 4. Configure

```bash
cp .env.example .env
```

Fill it in. `.env` is gitignored — keep it that way.

## Running

```bash
uv sync
uv run --env-file .env python bot.py
```

`--env-file` is what loads `.env`; nothing reads it implicitly, so plain `uv run python bot.py` starts with no configuration and exits.

Requires Python 3.14; `uv python install 3.14` if you don't have it.

## Using it

Send the bot a photo of food on the scale, with the display readable in frame.

| Command | Does |
|---|---|
| `/pending` | List items waiting on you, each with Approve / Discard buttons |
| `/today` | Show today's FatSecret diary |
| `/undo` | Remove the most recently logged entry |

Add a caption to skip the guessing: `Avocado` names the food, `56` sets the weight, `Avocado 56` does both. The caption wins over whatever the model reads.

A known food is logged immediately. A new one shows the top three FatSecret matches with their macros — tap one and it's logged and remembered. A weight unlike anything you've logged for that food is re-read once, then held.

Reply to any held item: a number fixes the weight, a name fixes the food.

Every reply shows the raw digits the model read alongside the parsed weight, so a misread is visible before it becomes a wrong calorie count.

## Deploying

```bash
docker build --target prod -t pht-food-bot:prod .
docker run -d --restart unless-stopped \
  --env-file .env \
  -v /srv/pht-food-bot/data:/app/data \
  pht-food-bot:prod
```

The bot uses long polling, so it needs no inbound port, no TLS certificate and no reverse proxy — outbound connections only.

**Mount the data volume.** `aliases.json`, `queue.json` and `log.jsonl` live there, and `aliases.json` is everything the bot has learned. Losing it means teaching every food again.

**Whitelist the VPS IP with FatSecret** before expecting it to work.

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy *.py         # strict type check
uv run pytest -q         # tests
```

Tests never touch the real Anthropic or FatSecret APIs — both are stubbed.

`docker build --target dev` gives an image with the dev tooling included.

## Layout

| File | Does |
|---|---|
| `models.py` | Shared types every module exchanges |
| `store.py` | JSON persistence with atomic writes, plus the audit log |
| `match.py` | Alias lookup and the gram→serving-units maths |
| `vision.py` | Photo → structured items, via Claude |
| `fatsecret.py` | OAuth1 client, food lookup, diary read/write, `authorize` CLI |
| `bot.py` | Telegram handlers and orchestration |


`SCOPE.md` defines commit conventions. `CLAUDE.md` holds the architectural notes and the gotchas that have already caused bugs.

## Status and known limits

- **Meals aren't split.** Everything logs to `other`; FatSecret requires a meal and the bot never picks one.
- **Whether Physical Life backfills older entries is unknown.** If it only syncs recent ones, the catch-up-on-a-week-of-backlog use case doesn't work and the bot becomes same-day only.
- The model defaults to `claude-haiku-4-5` on cost grounds. If it misreads scale digits, set `ANTHROPIC_MODEL=claude-sonnet-5` — check the `raw_digits` field on your first batch of real photos before deciding.
