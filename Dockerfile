# syntax=docker/dockerfile:1.9
#
# fatsecret_bot — three images from one file:
#
#   prod : slim; the six source modules and their runtime dependencies only.
#            docker buildx build --target prod -t fatsecret-bot .
#
#   dev  : adds pytest, ruff and mypy, the tests, and the benchmark. Shell in.
#            docker buildx build --target dev -t fatsecret-bot:dev .
#
#   test : runs the suite as its last step, so the build fails when a test does.
#            docker buildx build --target test .
#
# Needs BuildKit (DOCKER_BUILDKIT=1, or use `docker buildx`).
#
# No secret is ever built in. The bot reads its five keys from the environment
# at run time, so pass them then and keep .env on the host:
#
#   docker run -d --restart unless-stopped --env-file .env \
#     -v /srv/fatsecret-bot/data:/data fatsecret-bot
#
# .env and data/ are both excluded by .dockerignore. Baking either one would
# publish the owner's credentials and food diary to wherever the image goes.

ARG PYTHON_VERSION=3.14


FROM python:${PYTHON_VERSION}-slim AS builder
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# The environment lives outside /app so that copying source over it is
# impossible, and so prod can take it in a single COPY --from. PATH is set here
# rather than only in prod: without it a shell in the dev image gets the system
# interpreter instead of the project environment.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /app

# Lockfile first, so editing a module does not re-resolve the dependency tree.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Named individually rather than `COPY . .`: the build context also holds the
# audit log, the photo corpus and the tests, and none of them belong in prod.
COPY bot.py fatsecret.py match.py models.py store.py vision.py ./


FROM builder AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen
COPY tests ./tests
COPY benchmark.py ./
CMD ["/bin/bash"]


FROM dev AS test
RUN uv run ruff check . && uv run mypy *.py && uv run pytest -q


FROM python:${PYTHON_VERSION}-slim AS prod
# A fresh base, not the builder: prod inherits no uv binary, no compiler and no
# test tooling, only the environment and the modules that get imported.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app
# DATA_DIR is absolute and outside /app so the volume mount cannot shadow a
# source file. It holds aliases.json, queue.json, log.jsonl and photos/ — mount
# it, or the bot relearns every food on the next deploy.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data
WORKDIR /app
# Long polling only — the bot opens outbound connections and listens on no port.
CMD ["python", "bot.py"]
