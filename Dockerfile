# syntax=docker/dockerfile:1.9
FROM python:3.14-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# Bytecode compilation trades image build time for faster cold starts; copy link
# mode avoids hardlink failures when the cache mount and /app differ in device.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1
WORKDIR /app

# Runtime dependencies resolve in their own layer so editing source never re-runs
# the solve. Dev tooling is deliberately excluded here: a shared layer containing
# it would leave prod carrying the files in a lower layer and merely whiting them
# out later, making the prod image larger than dev.
FROM base AS deps
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

FROM deps AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen
CMD ["uv", "run", "python", "main.py"]

FROM deps AS prod
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
# Long polling only — the bot opens outbound connections and listens on no port.
CMD ["uv", "run", "--no-dev", "python", "main.py"]
