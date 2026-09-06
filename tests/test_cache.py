"""Tests for the prompt-hash cache.

The cache is what makes re-running a benchmark question free and makes
reported numbers reproducible, so its correctness properties are:
  - identical inputs hit; any changed input misses
  - a hit costs no quota
  - damage degrades to a miss, never to a crash
  - failures are never cached
"""
import json

import pytest

from src.llm.cache import CACHE_FORMAT_VERSION, CachedResponse, PromptCache

BASE = dict(
    model="m1", system="sys", user="usr",
    temperature=0.0, max_output_tokens=512, json_mode=False,
)


def resp(text="hello", total=150):
    return CachedResponse(
        text=text, model="m1", provider="groq",
        prompt_tokens=100, completion_tokens=50, total_tokens=total,
    )


@pytest.fixture
def cache(tmp_path):
    return PromptCache(tmp_path / "llm")


# --- keying ----------------------------------------------------------------


def test_identical_inputs_produce_identical_key():
    assert PromptCache.make_key(**BASE) == PromptCache.make_key(**BASE)


@pytest.mark.parametrize("field,value", [
    ("model", "m2"),
    ("system", "different"),
    ("user", "different"),
    ("temperature", 0.7),
    ("max_output_tokens", 1024),
    ("json_mode", True),
])
def test_every_field_changes_the_key(field, value):
    """Each of these changes the model's response, so each must change the
    key. Omitting one would serve a stale answer for a different request."""
    assert PromptCache.make_key(**{**BASE, field: value}) != PromptCache.make_key(**BASE)


def test_delimiter_prevents_field_boundary_collision():
    """('ab','c') and ('a','bc') must not hash alike."""
    a = PromptCache.make_key(**{**BASE, "system": "ab", "user": "c"})
    b = PromptCache.make_key(**{**BASE, "system": "a", "user": "bc"})
    assert a != b


# --- round trip ------------------------------------------------------------


def test_miss_then_hit(cache):
    key = PromptCache.make_key(**BASE)
    assert cache.get(key) is None
    cache.put(key, resp())
    got = cache.get(key)
    assert got is not None and got.text == "hello"
    assert cache.stats.hits == 1 and cache.stats.misses == 1


def test_hit_accumulates_tokens_saved(cache):
    key = PromptCache.make_key(**BASE)
    cache.put(key, resp(total=1200))
    cache.get(key)
    cache.get(key)
    assert cache.stats.tokens_saved == 2400


def test_hit_rate(cache):
    key = PromptCache.make_key(**BASE)
    cache.get(key)              # miss
    cache.put(key, resp())
    cache.get(key)              # hit
    assert cache.stats.hit_rate == 0.5


def test_entries_are_sharded(cache, tmp_path):
    key = PromptCache.make_key(**BASE)
    cache.put(key, resp())
    assert (tmp_path / "llm" / key[:2] / f"{key}.json").is_file()


def test_persists_across_instances(tmp_path):
    key = PromptCache.make_key(**BASE)
    PromptCache(tmp_path / "llm").put(key, resp())
    assert PromptCache(tmp_path / "llm").get(key).text == "hello"


# --- robustness ------------------------------------------------------------


def test_disabled_cache_never_reads_or_writes(tmp_path):
    cache = PromptCache(tmp_path / "llm", enabled=False)
    key = PromptCache.make_key(**BASE)
    cache.put(key, resp())
    assert cache.get(key) is None
    assert cache.count() == 0


def test_corrupt_entry_degrades_to_miss(cache, tmp_path):
    key = PromptCache.make_key(**BASE)
    cache.put(key, resp())
    (tmp_path / "llm" / key[:2] / f"{key}.json").write_text("{ not json")
    assert cache.get(key) is None
    assert cache.stats.errors == 1


def test_stale_format_version_is_a_miss(cache, tmp_path):
    key = PromptCache.make_key(**BASE)
    cache.put(key, resp())
    path = tmp_path / "llm" / key[:2] / f"{key}.json"
    data = json.loads(path.read_text())
    data["version"] = CACHE_FORMAT_VERSION + 1
    path.write_text(json.dumps(data))
    assert cache.get(key) is None


def test_clear_removes_entries(cache):
    for i in range(3):
        cache.put(PromptCache.make_key(**{**BASE, "user": f"u{i}"}), resp())
    assert cache.count() == 3
    assert cache.clear() == 3
    assert cache.count() == 0


def test_report_renders(cache):
    cache.put(PromptCache.make_key(**BASE), resp())
    assert "entries" in cache.report()