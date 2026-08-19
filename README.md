# fatsecret_bot

Photograph your food on kitchen scales, forward it to a Telegram chat, forget about it. The bot reads the product and the weight off the scale's display and logs it to your FatSecret diary.

## Why

Logging food by hand is slow—search, type, repeat for every item. Slow means deferred, deferred becomes a backlog, and a backlog becomes nothing logged at all.

So the design goal isn't "make typing faster," it's **a food you've eaten before takes zero interaction**. The first time you eat something you teach the bot what it is; every photo of it after that logs itself.

## How it works

```
photo → Gemini reads product + weight → known food? → FatSecret diary
                                             │
                                          new food → held for one tap
```

## Setup

Five steps. Each one ends with a value in `.env`, and the last one needs the others already there.

### Step 1: install

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/). If you don't have the interpreter:

```bash
uv python install 3.14
```

### Step 2: create your .env

```bash
cp .env.example .env
```

`.env` is gitignored—keep it that way. The rest of the setup fills it in.

### Step 3: Telegram

Message [@BotFather](https://t.me/BotFather) and send `/newbot`. Put the token in `TELEGRAM_BOT_TOKEN`.

Get your own numeric user id from [@userinfobot](https://t.me/userinfobot) and put it in `TELEGRAM_OWNER_ID`. The bot ignores everyone else.

### Step 4: Gemini

Create an API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and put it in `GEMINI_API_KEY`. There is a free tier.

### Step 5: FatSecret

1. Register at [`platform.fatsecret.com/register`](https://platform.fatsecret.com/register). The free **Basic** edition is self sign-up with no approval.
2. Under **Manage API Keys**, copy the **OAuth 1.0 Consumer Key and Shared Secret** into `FATSECRET_CONSUMER_KEY` and `FATSECRET_CONSUMER_SECRET`.

   > Not the OAuth 2.0 **Client ID / Client Secret**. FatSecret issues both, and only OAuth 1.0 reaches a user's diary. The wrong pair fails with a bare `400 Invalid signature`.

3. Under **IP Restrictions**, whitelist the machine that runs the bot. Basic allows 15 individual addresses and no CIDR ranges, so add your home IP now and the server IP when you deploy. Auth fails confusingly if you skip this.
4. Get an access token for your own account:

   ```bash
   uv run --env-file .env python fatsecret.py authorize
   ```

   It prints a URL. Approve it in a browser, and FatSecret shows a PIN. Paste the PIN back, and it prints a token and secret for `FATSECRET_ACCESS_TOKEN` and `FATSECRET_ACCESS_SECRET`. No callback server needed.

## Running

```bash
uv run --env-file .env python bot.py
```

`--env-file` is what loads `.env`. Nothing reads it implicitly, so plain `uv run python bot.py` starts with no configuration and exits.

## Using it

Send the bot a photo of food on the scale, with the display readable in frame. Then:

- **A food you've logged before** logs itself. You get `Logged Egg — 124 g` and nothing else to do.
- **A food it doesn't know** shows the top three FatSecret matches with their macros. Tap one, and the bot logs it and remembers the name for next time.
- **A weight unlike anything you've logged for that food** waits for you to confirm it, and shows the digits the model read. This is the check that catches a 56 g slice of bread read as 560 g.

### Correcting something

Reply to any message the bot sent you, whether it already logged the item or still holds it:

| Reply | Does |
|---|---|
| `118` | Sets the weight to 118 g |
| `-80` | Takes 80 g off, for rinds, bones and shells you can only weigh afterwards |
| `+50` | Adds 50 g |
| `Avocado` | Names the right food and remembers it |
| `Avocado 118` | Both |
| `y s` | Moves it to yesterday's snack |

For an item it already logged, the bot deletes the diary entry and writes a new one, so a correction never leaves two entries for one portion. If the correction can't work—subtracting more than the item weighs—the bot changes nothing and reports why.

### Captions

Add a caption to a photo to skip the guessing: `Avocado` names the food, `56` sets the weight, `Avocado 56` does both. A caption beats whatever the model reads, and `-80` works here too if you know the container's weight.

### Commands

| Command | Does |
|---|---|
| `/pending` | List items waiting on you, each with Approve / Discard buttons |
| `/today` | Show today's diary with macros and a daily total |
| `/undo` | Remove the most recently logged entry |
| `/help` | Prints the caption and reply grammar |
| `/new` | As a caption: read a product's nutrition label and add it to FatSecret |

**Catching up.** Send the photos as one batch and caption them with the day and meal, shortest form first:

| Caption | Files against |
|---|---|
| `y s` | Yesterday, snack |
| `t d` | Tomorrow, dinner |
| `15/08 l` | 15 August, lunch |
| `y l chicken 200 rice 150` | Yesterday's lunch: two foods with their weights |

`y` is yesterday and `t` is tomorrow. A written date is **day/month/year**, and `15/08` fills in the current year. Meals are `b`, `l`, `d` and `s`; a snack files as FatSecret's `other`, which its apps show as Snack. Leave either out and the bot keeps the photo's own day and the meal for that time of day. If it can't read the date it logs nothing at all, rather than filing the batch quietly against the wrong day.

The same grammar works as a reply, so `y s` on a logged item moves it. A line that says nothing about the day leaves the item where it is—a bare `-80` never drags a backlogged entry to today.

**Adding a product FatSecret doesn't have.** Photograph the package and its nutrition label, caption `/new`, and the bot reads the panel and shows it to you before writing anything. Name it yourself with `/new Super protein bar 23 g, SuperCompany`—everything before the last comma is the name, so a product that sells its macros in the title keeps them. The bot logs nothing here: the next scale photo of the product does that.

## Deploying

```bash
docker buildx build --target prod -t fatsecret-bot .
docker run -d --restart unless-stopped \
  --env-file .env \
  -v /srv/fatsecret-bot/data:/data \
  fatsecret-bot
```

The bot uses long polling, so it needs no inbound port, no TLS certificate and no reverse proxy—outbound connections only.

**Mount the data volume** at `/data`. `aliases.json`, `queue.json` and `log.jsonl` live there, and `aliases.json` is everything the bot has learned. Losing it means teaching every food again.

**Whitelist the server IP with FatSecret** before expecting it to work.

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy *.py         # strict type check
uv run pytest -q         # tests
```

Tests never call the real Gemini or FatSecret APIs. The suite stubs both.

`benchmark.py` does hit the real API: it replays every photo in `data/photos/` through a list of candidate models and scores them against the weights already in your diary. Run it before changing `GEMINI_MODEL`.

`docker buildx build --target dev` gives an image with the test and lint tooling included. `docker buildx build --target test` runs ruff, mypy and the suite inside the image and fails the build if any of them fails.

## Layout

| File | Does |
|---|---|
| `models.py` | Shared types every module exchanges |
| `store.py` | JSON persistence with atomic writes, plus the audit log |
| `match.py` | Alias lookup and the gram→serving-units maths |
| `vision.py` | Photo → structured items, via Gemini |
| `fatsecret.py` | OAuth1 client, food lookup, diary read/write, `authorize` command |
| `bot.py` | Telegram handlers and orchestration |

`SCOPES.md` defines commit conventions. `CLAUDE.md` holds the architectural notes and the gotchas that have already caused bugs.

## Status and known limits

- **You can't delete or edit a food once it exists.** FatSecret offers no `food.delete` and no `food.edit`, which is why `/new` shows the panel and waits for a tap. A wrong number means creating a corrected food and ignoring the first.
- **`/new` needs a Premier tier.** On the free Basic edition `food.create` returns `error 10, Unknown method`.
- **Don't turn on model thinking.** It triples latency, reads nothing better, and its tokens count against the output cap, so a photo can come back truncated with no answer. `CLAUDE.md` has the measurements.
