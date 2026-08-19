import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.api.backends import BackendManager, FakeBackend

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests/fixtures/fake_backend.json"


@pytest.fixture()
def client(tmp_path):
    main.cache = main.SimpleCache(str(tmp_path / "cache"))
    main.feedback_store = main.FeedbackStore(str(tmp_path / "feedback"))
    main.backend_manager = BackendManager("fake", str(FIXTURE_PATH))
    main.rate_limiter.clear()
    return TestClient(main.app)


def test_health_does_not_load_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["backend"] == "fake"
    assert payload["model_loaded"] is False


def test_analyze_returns_structured_fixture(client):
    question = "Explain the input validation controls in SI-10 to prevent invalid inputs."
    response = client.post("/analyze", json={"text": question})

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"].startswith("Fixture SI-10 guidance")
    assert payload["framework"] == "NIST SP 800-53 Rev. 5"
    assert payload["findings"][0]["control_id"] == "SI-10"
    assert payload["citations"][0]["locator"] == "SI-10"
    assert payload["cached"] is False

    cached = client.post("/analyze", json={"text": question})
    assert cached.status_code == 200
    assert cached.json()["cached"] is True
    assert cached.json()["text"] == payload["text"]


def test_generate_preserves_legacy_response_and_marks_deprecated(client):
    request = {"text": "Describe AC-2 account management."}
    response = client.post("/generate", json=request)

    assert response.status_code == 200
    payload = response.json()
    assert payload["generated_text"] == payload["text"]
    assert payload["framework"] == "NIST SP 800-53 Rev. 5"
    assert response.headers["deprecation"] == "true"
    assert response.headers["x-compliance-guard-deprecated"] == "true"

    cached = client.post("/generate", json=request)
    assert cached.status_code == 200
    assert cached.json()["cached"] is True
    assert cached.json()["generated_text"] == payload["generated_text"]


def test_legacy_compliance_adapter_keeps_query_contract(client):
    request = {"query": "What is AC-2?"}
    response = client.post("/compliance", json=request)

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == payload["text"]
    assert payload["source"] == payload["framework"]

    cached = client.post("/compliance", json=request)
    assert cached.status_code == 200
    assert cached.json()["cached"] is True
    assert cached.json()["answer"] == payload["answer"]


def test_cache_evicts_oldest_entry_and_preserves_non_cache_files(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = main.SimpleCache(str(cache_dir), max_entries=2, max_bytes=4096)
    value = {"text": "fixture", "framework": "NIST SP 800-53 Rev. 5"}

    cache.set("first", value)
    first_path = cache_dir / f"{cache._cache_key('first')}.json"
    cache.set("second", value)
    second_path = cache_dir / f"{cache._cache_key('second')}.json"
    os.utime(first_path, (1, 1))
    os.utime(second_path, (2, 2))
    unrelated = cache_dir / "keep-me.json"
    unrelated.write_text("outside cache filename", encoding="utf-8")

    cache.set("third", value)
    third_path = cache_dir / f"{cache._cache_key('third')}.json"

    assert not first_path.exists()
    assert second_path.exists()
    assert third_path.exists()
    assert unrelated.exists()
    assert len(list(cache_dir.glob("*.json"))) == 3


def test_cache_evicts_by_byte_limit(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = main.SimpleCache(str(cache_dir), max_entries=10, max_bytes=100)
    value = {"text": "x" * 60}

    cache.set("first-bytes", value)
    first_path = cache_dir / f"{cache._cache_key('first-bytes')}.json"
    os.utime(first_path, (1, 1))
    cache.set("second-bytes", value)
    second_path = cache_dir / f"{cache._cache_key('second-bytes')}.json"

    assert not first_path.exists()
    assert second_path.exists()
    assert sum(path.stat().st_size for path in cache_dir.glob("*.json")) <= 100


def test_body_limit_is_enforced_before_validation(client):
    response = client.post(
        "/analyze",
        content=b"x" * (main.MAX_API_BODY_BYTES + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413


def test_analysis_rate_limit_is_enforced(client):
    for index in range(main.RATE_LIMIT):
        response = client.post("/analyze", json={"text": f"rate-limit-{index}"})
        assert response.status_code == 200

    limited = client.post("/analyze", json={"text": "rate-limit-overflow"})
    assert limited.status_code == 429
    assert "retry-after" in limited.headers


def test_inputs_are_not_rejected_by_keyword_blacklist(client):
    response = client.post(
        "/analyze", json={"text": "Discuss system instructions as audit evidence."}
    )

    assert response.status_code == 200


def test_golden_set_is_deterministic_and_structured():
    backend = FakeBackend(str(FIXTURE_PATH))
    cases = json.loads((ROOT / "tests/golden_set.json").read_text(encoding="utf-8"))

    assert cases
    for case in cases:
        first = backend.analyze(case["question"]).as_dict()
        second = backend.analyze(case["question"]).as_dict()
        assert first == second
        assert first["text"]
        assert first["framework"]
        assert isinstance(first["findings"], list)
        assert isinstance(first["citations"], list)
        expected_keywords = [keyword.casefold() for keyword in case.get("expected_keywords", [])]
        assert any(keyword in first["text"].casefold() for keyword in expected_keywords)
