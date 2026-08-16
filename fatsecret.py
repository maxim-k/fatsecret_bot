"""FatSecret diary client: OAuth 1.0 signing and the REST calls the bot needs.

Only the operations the bot actually performs are implemented: searching
foods, reading a food's servings, listing a day's diary, and creating or
deleting entries. FatSecret's Basic tier has no OAuth2 option for
account-writing endpoints, so three-legged OAuth 1.0 is unavoidable here;
the ``authorize`` CLI at the bottom exists to obtain the access token once,
out of band, without standing up a callback server.

There is deliberately no client class or connection pool: a fresh
``httpx.AsyncClient`` is opened per call. This is a single-user bot making a
handful of calls a day, well under the free tier's 5,000/day cap, so the
extra TCP handshake is not worth the state management it would save.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import sys
import time
import uuid
from base64 import b64encode
from datetime import date
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode

import httpx

from models import DiaryEntry, Food, Serving, to_epoch_days

_AUTH_BASE = "https://authentication.fatsecret.com/oauth"
_REQUEST_TOKEN_URL = f"{_AUTH_BASE}/request_token"
_AUTHORIZE_URL = f"{_AUTH_BASE}/authorize"
_ACCESS_TOKEN_URL = f"{_AUTH_BASE}/access_token"

# The classic method-dispatch API: still the documented, stable way to reach
# foods.search and food.get.v2 on the Basic tier.
_API_URL = "https://platform.fatsecret.com/rest/server.api"
# The newer REST-style diary resource, dispatched by HTTP verb rather than a
# "method" parameter. Verified fact from the task brief: POST here writes a
# diary entry.
_DIARY_URL = "https://platform.fatsecret.com/rest/food-entries/v1"

_OAUTH_SIGNATURE_METHOD = "HMAC-SHA1"


class FatSecretError(Exception):
    """Raised when FatSecret's API returns an ``error`` object.

    FatSecret answers protocol-level failures (bad food id, expired token,
    invalid parameter) with HTTP 200 and an ``{"error": {...}}`` body rather
    than a non-2xx status, so callers cannot rely on ``raise_for_status``
    alone to detect failure.
    """

    def __init__(self, code: int | str, message: str) -> None:
        """Store the FatSecret error code and message.

        :param code: The ``error.code`` field from the response body.
        :param message: The ``error.message`` field from the response body.
        """
        self.code = code
        self.message = message
        super().__init__(f"FatSecret API error {code}: {message}")


def _percent_encode(value: str) -> str:
    """Percent-encode per RFC 3986 / OAuth 1.0 (RFC 5849 section 3.6).

    ``urllib.parse.quote``'s default safe set keeps ``/``, which OAuth1
    requires to be encoded too, so the safe set must be overridden to empty.
    Getting this wrong is the single most common OAuth1 signing bug.

    :param value: Raw string to encode.
    :returns: Percent-encoded string safe for use in a signature base string.
    """
    return quote(str(value), safe="~")


def signature_base_string(method: str, url: str, params: dict[str, str]) -> str:
    """Build the OAuth1 signature base string per RFC 5849 section 3.4.1.

    Parameters are sorted by their percent-encoded key/value before joining,
    and the whole parameter string is then percent-encoded a second time
    alongside the method and URL. An unsorted or single-encoded base string
    produces a signature FatSecret silently rejects.

    :param method: HTTP method, e.g. ``"POST"``.
    :param url: Base URL without query string.
    :param params: All OAuth and request parameters to sign, unencoded.
    :returns: The base string ready to be HMAC-signed.
    """
    encoded_params = sorted(
        (_percent_encode(key), _percent_encode(value)) for key, value in params.items()
    )
    param_string = "&".join(f"{key}={value}" for key, value in encoded_params)
    return "&".join(
        [_percent_encode(method.upper()), _percent_encode(url), _percent_encode(param_string)]
    )


def _sign(
    method: str,
    url: str,
    params: dict[str, str],
    consumer_secret: str,
    token_secret: str = "",
) -> str:
    """Compute the HMAC-SHA1 OAuth1 signature for a request.

    :param method: HTTP method, e.g. ``"POST"``.
    :param url: Base URL without query string.
    :param params: All OAuth and request parameters to sign, unencoded.
    :param consumer_secret: The app's consumer secret.
    :param token_secret: The token secret; empty during the request-token step.
    :returns: Base64-encoded signature.
    """
    base_string = signature_base_string(method, url, params)
    signing_key = f"{_percent_encode(consumer_secret)}&{_percent_encode(token_secret)}"
    digest = hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
    return b64encode(digest).decode()


def _oauth_params(consumer_key: str, token: str | None) -> dict[str, str]:
    """Build the standard OAuth1 parameter set common to every request.

    :param consumer_key: The app's consumer key.
    :param token: The access/request token, or ``None`` before one exists.
    :returns: OAuth parameters, excluding the signature itself.
    """
    params = {
        "oauth_consumer_key": consumer_key,
        "oauth_nonce": uuid.uuid4().hex,
        "oauth_signature_method": _OAUTH_SIGNATURE_METHOD,
        "oauth_timestamp": str(int(time.time())),
        "oauth_version": "1.0",
    }
    if token is not None:
        params["oauth_token"] = token
    return params


def _as_list(value: object) -> list[Any]:
    """Normalise FatSecret's object-or-list inconsistency to always a list.

    FatSecret's JSON translation of its underlying XML collapses a
    single-child collection to a bare object and only uses a JSON list when
    there are multiple children; a missing collection is sometimes just
    absent (``None``). Every caller that walks a nested collection hits
    this, so it is centralised here rather than repeated ad hoc at each
    call site.

    :param value: The raw field value: ``None``, a ``dict``, or a ``list``.
    :returns: Always a list, possibly empty.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _num(value: object) -> float:
    """Coerce a FatSecret numeric field to ``float``.

    FatSecret's JSON encodes numbers as strings ("31.50"), a consequence of
    the same XML-to-JSON translation behind the object-vs-list quirk.

    :param value: The raw field value, typically a ``str``.
    :returns: The parsed float.
    """
    return float(value)  # type: ignore[arg-type]


async def _request(method: str, url: str, params: dict[str, str]) -> dict[str, Any]:
    """Sign and send a request, returning the decoded JSON body.

    :param method: HTTP method, e.g. ``"POST"``, ``"GET"``, ``"DELETE"``.
    :param url: Full base URL (no query string).
    :param params: Request parameters, excluding OAuth boilerplate.
    :returns: Decoded JSON response body.
    :raises FatSecretError: If the response body contains an ``error`` object.
    """
    # KeyError already names the missing variable, so no bespoke check.
    env = os.environ
    all_params = {
        **params,
        **_oauth_params(env["FATSECRET_CONSUMER_KEY"], env["FATSECRET_ACCESS_TOKEN"]),
    }
    all_params["oauth_signature"] = _sign(
        method,
        url,
        all_params,
        env["FATSECRET_CONSUMER_SECRET"],
        env["FATSECRET_ACCESS_SECRET"],
    )
    async with httpx.AsyncClient() as client:
        if method in ("GET", "DELETE"):
            response = await client.request(method, url, params=all_params)
        else:
            response = await client.request(method, url, data=all_params)
    response.raise_for_status()
    body: dict[str, Any] = response.json()
    # FatSecret reports invalid parameters, expired tokens, unknown food ids,
    # etc. as HTTP 200 with an "error" object, not as a 4xx/5xx status, so a
    # successful raise_for_status() above says nothing about whether the
    # call actually succeeded — the body must be inspected explicitly.
    if "error" in body:
        error = body["error"]
        raise FatSecretError(error.get("code", "unknown"), error.get("message", ""))
    return body


async def search_foods(query: str) -> list[Food]:
    """Search FatSecret's food database by free-text query.

    :param query: Free-text search string, e.g. a product name.
    :returns: Matching foods. ``servings`` is empty on each result; call
        :func:`get_food` for a specific food's servings.
    :raises FatSecretError: If FatSecret returns an error response.
    """
    body = await _request(
        "POST",
        _API_URL,
        {"method": "foods.search", "search_expression": query, "format": "json"},
    )
    raw_foods = _as_list((body.get("foods") or {}).get("food"))
    return [
        Food(food_id=int(food["food_id"]), name=food["food_name"], servings=[])
        for food in raw_foods
    ]


async def get_food(food_id: int) -> Food:
    """Fetch a single food, including all of its servings.

    :param food_id: FatSecret's numeric food id.
    :returns: The food with ``servings`` fully populated.
    :raises FatSecretError: If FatSecret returns an error response.
    """
    body = await _request(
        "POST", _API_URL, {"method": "food.get.v2", "food_id": str(food_id), "format": "json"}
    )
    raw_food = body["food"]
    raw_servings = _as_list((raw_food.get("servings") or {}).get("serving"))
    servings = [_parse_serving(raw_serving) for raw_serving in raw_servings]
    return Food(food_id=int(raw_food["food_id"]), name=raw_food["food_name"], servings=servings)


def _parse_serving(raw: dict[str, Any]) -> Serving:
    """Parse one raw serving object into a :class:`models.Serving`.

    :param raw: One element of the ``food.servings.serving`` field.
    :returns: The parsed serving.
    """
    metric_amount = raw.get("metric_serving_amount")
    return Serving(
        serving_id=int(raw["serving_id"]),
        description=raw["serving_description"],
        metric_amount=_num(metric_amount) if metric_amount is not None else None,
        metric_unit=raw.get("metric_serving_unit"),
    )


async def list_diary(day: date) -> list[DiaryEntry]:
    """List diary entries logged on a given day.

    :param day: Calendar date to read back.
    :returns: Entries logged on that day, empty if none.
    :raises FatSecretError: If FatSecret returns an error response.
    """
    # See models.to_epoch_days: FatSecret counts diary days from 1970-01-01
    # rather than using a timestamp, so the calendar date is converted at
    # this boundary and nowhere else in the codebase needs to know about it.
    epoch_day = to_epoch_days(day)
    body = await _request("GET", _DIARY_URL, {"date": str(epoch_day), "format": "json"})
    raw_entries = _as_list((body.get("food_entries") or {}).get("food_entry"))
    return [_parse_diary_entry(raw_entry, day) for raw_entry in raw_entries]


def _parse_diary_entry(raw: dict[str, Any], day: date) -> DiaryEntry:
    """Parse one raw diary entry object into a :class:`models.DiaryEntry`.

    :param raw: One element of the ``food_entries.food_entry`` field.
    :param day: The day that was requested; the entries are all from it.
    :returns: The parsed entry.
    """
    return DiaryEntry(
        food_entry_id=int(raw["food_entry_id"]),
        food_id=int(raw["food_id"]),
        name=raw["food_entry_name"],
        day=day,
    )


async def create_entry(
    food_id: int, name: str, serving_id: int, units: float, day: date, meal: str = "other"
) -> DiaryEntry:
    """Log a food entry to the diary.

    :param food_id: FatSecret's numeric food id.
    :param name: Display name to store against the entry.
    :param serving_id: The serving option being logged.
    :param units: Number of servings, e.g. ``1.5``.
    :param day: Calendar date to log against.
    :param meal: Diary section; FatSecret requires one and the bot never splits meals.
    :returns: The newly created diary entry.
    :raises FatSecretError: If FatSecret returns an error response.
    """
    epoch_day = to_epoch_days(day)  # see models.to_epoch_days
    body = await _request(
        "POST",
        _DIARY_URL,
        {
            "food_id": str(food_id),
            "food_entry_name": name,
            "serving_id": str(serving_id),
            "number_of_units": str(units),
            "meal": meal,
            "date": str(epoch_day),
            "format": "json",
        },
    )
    raw_id = body["food_entry_id"]
    entry_id = int(raw_id["value"]) if isinstance(raw_id, dict) else int(raw_id)
    return DiaryEntry(food_entry_id=entry_id, food_id=food_id, name=name, day=day)


async def delete_entry(food_entry_id: int) -> None:
    """Delete a diary entry.

    :param food_entry_id: The entry id previously returned by
        :func:`create_entry` or :func:`list_diary`.
    :raises FatSecretError: If FatSecret returns an error response.
    """
    await _request("DELETE", _DIARY_URL, {"food_entry_id": str(food_entry_id), "format": "json"})


async def _fetch_request_token(
    client: httpx.AsyncClient, consumer_key: str, consumer_secret: str
) -> tuple[str, str]:
    """Fetch an unauthorised request token to start the three-legged flow.

    :param client: HTTP client to issue the request with.
    :param consumer_key: The app's OAuth1 consumer key.
    :param consumer_secret: The app's OAuth1 consumer secret.
    :returns: ``(oauth_token, oauth_token_secret)``.
    :raises FatSecretError: If FatSecret reports an ``oauth_problem``.
    """
    params = _oauth_params(consumer_key, None)
    params["oauth_callback"] = "oob"
    params["oauth_signature"] = _sign("POST", _REQUEST_TOKEN_URL, params, consumer_secret)
    response = await client.post(_REQUEST_TOKEN_URL, data=params)
    # raise_for_status() discards the response body, but FatSecret puts the
    # actual reason there ("Invalid signature", "Invalid consumer key") while
    # returning a bare 400 — so surface it or every failure looks identical.
    if response.is_error:
        raise FatSecretError(str(response.status_code), response.text.strip())
    body = dict(parse_qsl(response.text))
    if "oauth_problem" in body:
        raise FatSecretError("oauth_problem", body["oauth_problem"])
    return body["oauth_token"], body["oauth_token_secret"]


async def _fetch_access_token(
    client: httpx.AsyncClient,
    consumer_key: str,
    consumer_secret: str,
    request_token: str,
    request_secret: str,
    verifier: str,
) -> tuple[str, str]:
    """Exchange an authorised request token and PIN for an access token.

    :param client: HTTP client to issue the request with.
    :param consumer_key: The app's OAuth1 consumer key.
    :param consumer_secret: The app's OAuth1 consumer secret.
    :param request_token: The token obtained from :func:`_fetch_request_token`.
    :param request_secret: The secret paired with ``request_token``.
    :param verifier: The PIN FatSecret displayed after the user authorized.
    :returns: ``(oauth_token, oauth_token_secret)`` — the durable access token.
    :raises FatSecretError: If FatSecret reports an ``oauth_problem``.
    """
    params = _oauth_params(consumer_key, request_token)
    params["oauth_verifier"] = verifier
    params["oauth_signature"] = _sign(
        "POST", _ACCESS_TOKEN_URL, params, consumer_secret, request_secret
    )
    response = await client.post(_ACCESS_TOKEN_URL, data=params)
    # raise_for_status() discards the response body, but FatSecret puts the
    # actual reason there ("Invalid signature", "Invalid consumer key") while
    # returning a bare 400 — so surface it or every failure looks identical.
    if response.is_error:
        raise FatSecretError(str(response.status_code), response.text.strip())
    body = dict(parse_qsl(response.text))
    if "oauth_problem" in body:
        raise FatSecretError("oauth_problem", body["oauth_problem"])
    return body["oauth_token"], body["oauth_token_secret"]


async def _run_authorize_flow() -> None:
    """Walk the three-legged ``oob`` OAuth1 flow interactively on the CLI.

    Reads ``FATSECRET_CONSUMER_KEY``/``FATSECRET_CONSUMER_SECRET`` from the
    environment, prints the authorize URL, waits for the user to paste back
    the PIN FatSecret displays, and prints the resulting access token and
    secret so they can be copied straight into the bot's env.

    :raises SystemExit: If the consumer key/secret are not set.
    """
    consumer_key = os.environ.get("FATSECRET_CONSUMER_KEY")
    consumer_secret = os.environ.get("FATSECRET_CONSUMER_SECRET")
    if not consumer_key or not consumer_secret:
        print(
            "Set FATSECRET_CONSUMER_KEY and FATSECRET_CONSUMER_SECRET before running 'authorize'.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    async with httpx.AsyncClient() as client:
        request_token, request_secret = await _fetch_request_token(
            client, consumer_key, consumer_secret
        )
        authorize_url = f"{_AUTHORIZE_URL}?{urlencode({'oauth_token': request_token})}"
        print("Open this URL, log in, and authorize the app:")
        print(f"  {authorize_url}")
        # input() blocks the whole event loop; running it in a thread keeps
        # this coroutine well-behaved even though nothing else is competing
        # for the loop in a one-shot CLI script.
        pin = (await asyncio.to_thread(input, "Paste the PIN FatSecret displays: ")).strip()
        access_token, access_secret = await _fetch_access_token(
            client, consumer_key, consumer_secret, request_token, request_secret, pin
        )

    print()
    print("Add these to your environment:")
    print(f"  FATSECRET_ACCESS_TOKEN={access_token}")
    print(f"  FATSECRET_ACCESS_SECRET={access_secret}")


def _main() -> None:
    """CLI entry point: ``python fatsecret.py authorize``.

    :raises SystemExit: On unknown or missing subcommand.
    """
    if len(sys.argv) != 2 or sys.argv[1] != "authorize":
        print("Usage: python fatsecret.py authorize", file=sys.stderr)
        raise SystemExit(1)
    asyncio.run(_run_authorize_flow())


if __name__ == "__main__":
    _main()
