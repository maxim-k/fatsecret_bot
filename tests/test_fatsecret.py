from __future__ import annotations

import time
from datetime import date
from urllib.parse import parse_qs

import httpx
import pytest
import respx

import fatsecret
from fatsecret import (
    FatSecretError,
    _as_list,
    _num,
    create_entry,
    delete_entry,
    get_food,
    list_diary,
    search_foods,
    signature_base_string,
)
from models import to_epoch_days

# --- env fixture -------------------------------------------------------


@pytest.fixture(autouse=True)
def _creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FATSECRET_CONSUMER_KEY", "test_key")
    monkeypatch.setenv("FATSECRET_CONSUMER_SECRET", "test_secret")
    monkeypatch.setenv("FATSECRET_ACCESS_TOKEN", "test_token")
    monkeypatch.setenv("FATSECRET_ACCESS_SECRET", "test_token_secret")
    # The live burst guard would otherwise add 1.5 s to every test.
    monkeypatch.setattr(fatsecret, "_MIN_CALL_INTERVAL", 0.0)
    monkeypatch.setattr(fatsecret, "_last_call", 0.0)


def _body_params(request: httpx.Request) -> dict[str, str]:
    parsed = parse_qs(request.content.decode())
    return {k: v[0] for k, v in parsed.items()}


def _query_params(request: httpx.Request) -> dict[str, str]:
    parsed = parse_qs(request.url.query.decode())
    return {k: v[0] for k, v in parsed.items()}


# --- signature base string --------------------------------------------


def test_signature_base_string_sorted_and_encoded() -> None:
    # Deliberately out of alphabetical order and containing characters (":",
    # "/", "=", "&") that must be percent-encoded, to catch both the sorting
    # and the encoding halves of RFC 5849 section 3.4.1 at once.
    params = {"b": "2", "a": "1", "oauth_nonce": "abc123"}
    base = signature_base_string("get", "https://example.com/resource", params)
    assert base == ("GET&https%3A%2F%2Fexample.com%2Fresource&a%3D1%26b%3D2%26oauth_nonce%3Dabc123")


def test_signature_base_string_encodes_reserved_chars_in_values() -> None:
    base = signature_base_string("POST", "https://example.com/r", {"q": "hello world/x+y"})
    # The param string is percent-encoded once to build "q=...", then the
    # whole base string is percent-encoded again per RFC 5849 3.4.1.1 — so a
    # space ends up "%2520" (%20 re-encoded), not "%20" or "+". Space must
    # never become "+" (that's quote_plus/urlencode behaviour, not OAuth1's),
    # and "/" inside a value must be encoded too — both are easy to get
    # wrong with urlencode()/quote_plus.
    assert "hello%2520world%252Fx%252By" in base
    assert "+" not in base


# --- _as_list / _num normalisation --------------------------------------


def test_as_list_none_becomes_empty() -> None:
    assert _as_list(None) == []


def test_as_list_single_object_becomes_singleton_list() -> None:
    obj = {"food_id": "1"}
    assert _as_list(obj) == [obj]


def test_as_list_passes_through_existing_list() -> None:
    items = [{"food_id": "1"}, {"food_id": "2"}]
    assert _as_list(items) is items


def test_num_coerces_string() -> None:
    assert _num("31.50") == pytest.approx(31.5)


def test_num_passes_through_numeric() -> None:
    assert _num(12) == pytest.approx(12.0)


# --- error-as-200 handling ----------------------------------------------


@respx.mock
async def test_error_object_with_http_200_raises() -> None:
    respx.post(fatsecret._API_URL).mock(
        return_value=httpx.Response(
            200, json={"error": {"code": 106, "message": "IP is not authorized"}}
        )
    )
    with pytest.raises(FatSecretError) as exc_info:
        await search_foods("egg")
    assert exc_info.value.code == 106
    assert "not authorized" in exc_info.value.message


# --- search_foods --------------------------------------------------------


@respx.mock
async def test_search_foods_normalizes_single_result_object() -> None:
    respx.post(fatsecret._API_URL).mock(
        return_value=httpx.Response(
            200, json={"foods": {"food": {"food_id": "12345", "food_name": "Egg"}}}
        )
    )
    results = await search_foods("egg")
    assert len(results) == 1
    assert results[0].food_id == 12345
    assert results[0].name == "Egg"
    assert results[0].servings == []


@respx.mock
async def test_search_foods_keeps_multiple_results_as_list() -> None:
    respx.post(fatsecret._API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "foods": {
                    "food": [
                        {"food_id": "1", "food_name": "Egg, raw"},
                        {"food_id": "2", "food_name": "Egg, boiled"},
                    ]
                }
            },
        )
    )
    results = await search_foods("egg")
    assert [f.food_id for f in results] == [1, 2]


@respx.mock
async def test_search_foods_no_results() -> None:
    respx.post(fatsecret._API_URL).mock(return_value=httpx.Response(200, json={"foods": {}}))
    assert await search_foods("nonexistent") == []


# --- get_food / serving parsing ------------------------------------------


@respx.mock
async def test_get_food_parses_servings_and_coerces_numbers() -> None:
    respx.post(fatsecret._API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "food": {
                    "food_id": "33691",
                    "food_name": "Egg",
                    "servings": {
                        "serving": [
                            {
                                "serving_id": "1",
                                "serving_description": "1 large",
                                "metric_serving_amount": "50.000",
                                "metric_serving_unit": "g",
                            },
                            {
                                "serving_id": "2",
                                "serving_description": "1 bar",
                            },
                        ]
                    },
                }
            },
        )
    )
    food = await get_food(33691)
    assert food.food_id == 33691
    assert food.name == "Egg"
    assert len(food.servings) == 2

    gram_serving = food.servings[0]
    assert gram_serving.serving_id == 1
    assert gram_serving.metric_amount == pytest.approx(50.0)
    assert isinstance(gram_serving.metric_amount, float)
    assert gram_serving.metric_unit == "g"

    natural_serving = food.servings[1]
    assert natural_serving.metric_amount is None
    assert natural_serving.metric_unit is None


@respx.mock
async def test_get_food_single_serving_normalized_to_list() -> None:
    respx.post(fatsecret._API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "food": {
                    "food_id": "1",
                    "food_name": "Egg",
                    "servings": {
                        "serving": {
                            "serving_id": "1",
                            "serving_description": "1 large",
                            "metric_serving_amount": "50",
                            "metric_serving_unit": "g",
                        }
                    },
                }
            },
        )
    )
    food = await get_food(1)
    assert len(food.servings) == 1
    assert food.servings[0].serving_id == 1


# --- list_diary ------------------------------------------------------------


@respx.mock
async def test_list_diary_normalizes_single_entry_and_dates() -> None:
    day = date(2026, 8, 13)
    respx.get(fatsecret._DIARY_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "food_entries": {
                    "food_entry": {
                        "food_entry_id": "999",
                        "food_id": "42",
                        "food_entry_name": "Egg",
                        "date_int": str(to_epoch_days(day)),
                    }
                }
            },
        )
    )
    entries = await list_diary(day)
    assert len(entries) == 1
    assert entries[0].food_entry_id == 999
    assert entries[0].food_id == 42
    assert entries[0].day == day


@respx.mock
async def test_list_diary_sends_epoch_day_not_calendar_date() -> None:
    day = date(2026, 8, 13)
    route = respx.get(fatsecret._DIARY_URL).mock(
        return_value=httpx.Response(200, json={"food_entries": {}})
    )
    await list_diary(day)
    sent = _query_params(route.calls.last.request)
    assert sent["date"] == str(to_epoch_days(day))
    # Guard against accidentally sending a timestamp or ISO date instead.
    assert sent["date"] != day.isoformat()


# --- create_entry ------------------------------------------------------------


@respx.mock
async def test_create_entry_sends_correct_epoch_day_and_fields() -> None:
    day = date(2026, 8, 13)
    route = respx.post(fatsecret._DIARY_URL).mock(
        return_value=httpx.Response(200, json={"food_entry_id": "555"})
    )
    entry = await create_entry(
        food_id=42,
        name="Egg",
        serving_id=1,
        units=1.5,
        meal="breakfast",
        day=day,
    )
    sent = _body_params(route.calls.last.request)
    assert sent["date"] == str(to_epoch_days(day))
    assert sent["food_id"] == "42"
    assert sent["serving_id"] == "1"
    assert sent["number_of_units"] == "1.5"
    assert sent["meal"] == "breakfast"
    assert entry.food_entry_id == 555
    assert entry.day == day


@respx.mock
async def test_create_entry_unwraps_value_wrapped_id() -> None:
    respx.post(fatsecret._DIARY_URL).mock(
        return_value=httpx.Response(200, json={"food_entry_id": {"value": "777"}})
    )
    entry = await create_entry(
        food_id=1, name="Egg", serving_id=1, units=1.0, day=date(2026, 8, 13)
    )
    assert entry.food_entry_id == 777


# --- delete_entry ------------------------------------------------------------


@respx.mock
async def test_delete_entry_sends_entry_id_and_returns_none() -> None:
    route = respx.delete(fatsecret._DIARY_URL).mock(return_value=httpx.Response(200, json={}))
    result = await delete_entry(555)
    assert result is None
    sent = _query_params(route.calls.last.request)
    assert sent["food_entry_id"] == "555"


# --- credentials -----------------------------------------------------------


async def test_missing_credentials_raise_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FATSECRET_CONSUMER_KEY", raising=False)
    # KeyError from os.environ already names the missing variable.
    with pytest.raises(KeyError, match="FATSECRET_CONSUMER_KEY"):
        await search_foods("egg")


# --- sanity: epoch day round trip used throughout ---------------------------


@pytest.mark.asyncio
@respx.mock
async def test_list_diary_handles_null_collection_on_an_empty_day() -> None:
    # An empty day comes back with the key PRESENT and null, so `.get(k, {})`
    # returns None and blows up. Every day with no food logged hit this.
    respx.get(url__startswith=fatsecret._DIARY_URL).mock(
        return_value=httpx.Response(200, json={"food_entries": None})
    )
    assert await fatsecret.list_diary(date(2026, 8, 15)) == []


@pytest.mark.asyncio
@respx.mock
async def test_search_foods_handles_null_collection() -> None:
    respx.post(fatsecret._API_URL).mock(return_value=httpx.Response(200, json={"foods": None}))
    assert await fatsecret.search_foods("nothing matches this") == []


@pytest.mark.asyncio
@respx.mock
async def test_calls_are_spaced_to_survive_a_bulk_album(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A Telegram album fires one update per photo; back-to-back submits used to
    # trip FatSecret's error 12 and drop most of the batch.
    monkeypatch.setattr(fatsecret, "_MIN_CALL_INTERVAL", 0.05)
    respx.post(fatsecret._API_URL).mock(return_value=httpx.Response(200, json={"foods": None}))
    start = time.monotonic()
    for _ in range(3):
        await search_foods("egg")
    elapsed = time.monotonic() - start
    assert elapsed >= 2 * fatsecret._MIN_CALL_INTERVAL


@pytest.mark.asyncio
@respx.mock
async def test_search_keeps_macros_and_brand_for_the_picker() -> None:
    respx.post(fatsecret._API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "foods": {
                    "food": [
                        {
                            "food_id": "66325",
                            "food_name": "Avocado",
                            "brand_name": "Calavo",
                            "food_description": "Per 1 avocado - Calories: 227kcal",
                        },
                        {"food_id": "35752", "food_name": "Avocados"},
                    ]
                }
            },
        )
    )
    branded, generic = await search_foods("avocado")
    assert branded.brand == "Calavo"
    assert branded.description is not None
    assert "227kcal" in branded.description
    # Absent fields stay None rather than becoming empty strings.
    assert generic.brand is None
    assert generic.description is None


@respx.mock
async def test_create_food_sends_the_panel_and_returns_the_id() -> None:
    route = respx.post("https://platform.fatsecret.com/rest/server.api").mock(
        return_value=httpx.Response(200, json={"food_id": {"value": "12345"}})
    )
    food_id = await fatsecret.create_food(
        name="Super protein bar 23 g",
        brand="SuperCompany",
        serving_description="1 bar",
        nutrients={"calories": 240.0, "fat": 11.0, "carbohydrate": 24.0, "protein": 23.0},
        serving_amount=60.0,
    )
    assert food_id == 12345
    sent = _body_params(route.calls[0].request)
    assert sent["method"] == "food.create.v2"
    assert sent["food_name"] == "Super protein bar 23 g"
    assert sent["brand_name"] == "SuperCompany"
    assert sent["brand_type"] == "manufacturer"
    assert sent["serving_size"] == "1 bar"
    assert sent["serving_amount"] == "60.0"
    assert sent["serving_amount_unit"] == "g"
    assert sent["protein"] == "23.0"


@respx.mock
async def test_create_food_omits_the_serving_amount_when_the_label_gave_none() -> None:
    route = respx.post("https://platform.fatsecret.com/rest/server.api").mock(
        return_value=httpx.Response(200, json={"food_id": "999"})
    )
    assert await fatsecret.create_food("Bar", "Brand", "1 bar", {"calories": 100.0}) == 999
    sent = _body_params(route.calls[0].request)
    assert "serving_amount" not in sent


@respx.mock
async def test_create_food_surfaces_a_tier_rejection() -> None:
    # Basic answers error 10 for this method; the caller must see that rather
    # than a bare KeyError on the missing food_id.
    respx.post("https://platform.fatsecret.com/rest/server.api").mock(
        return_value=httpx.Response(200, json={"error": {"code": 10, "message": "Unknown method"}})
    )
    with pytest.raises(fatsecret.FatSecretError) as caught:
        await fatsecret.create_food("Bar", "Brand", "1 bar", {"calories": 100.0})
    assert caught.value.code == 10
