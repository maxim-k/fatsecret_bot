"""Entry point.

Long polling rather than webhooks: the bot serves exactly one user, so there is
nothing to gain from a public endpoint and a lot to avoid — no inbound port, no
TLS certificate, no reverse proxy.
"""

from __future__ import annotations

import logging
import os
import sys

import bot


def main() -> int:
    """Start the bot and poll until interrupted.

    :returns: Process exit code.
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Telegram's own HTTP chatter buries our logs at INFO and is rarely useful.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logging.error("TELEGRAM_BOT_TOKEN is not set — see .env.example")
        return 1

    bot.build_application(token).run_polling()
    return 0


if __name__ == "__main__":
    sys.exit(main())
