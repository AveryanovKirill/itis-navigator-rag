import asyncio
import csv
import json
import pytest
from src.agent import ask, OpenRouter, APIError
from src.schemas import REFUSAL
from src.evaluation import aggregate, export_review, error_analysis
from src.reporting import cost_projection
from src.storage import save_json


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    async def complete(self, messages, checkpoint=None):
        self.messages.append(json.loads(json.dumps(messages)))
        return {"response": {"model": "TEST_ONLY", "choices": [{"message": {"content": next(self.outputs)}}],
                             "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}}


def test_retry_fallback_counts_all_tokens():
    fake = FakeLLM(["bad", "bad"])
    result = asyncio.run(ask("Когда экзамен?", "v1", fake))
    assert result["origin"] == "validation_fallback"
    assert result["answer"]["answer"] == REFUSAL
    assert result["usage"]["prompt_tokens"] == 200
    assert result["retrieval"] is None
    assert "search_docs" not in fake.messages[0][1]["content"]


def test_v2_gets_mcp_not_gold(monkeypatch):
    import src.agent
    calls = []
    async def fake_mcp(session, question):
        calls.append(question)
        return {"status": "found", "fragments": [{"number": 1, "text": "Старт 10:00."}]}
    monkeypatch.setattr(src.agent, "call_search", fake_mcp)
    fake = FakeLLM(['{"answer":"Старт 10:00 [1].","sources":[1],"found":true}'])
    result = asyncio.run(ask("Когда начало?", "v2", fake, mcp=object()))
    assert result["origin"] == "gemini"  # Только unit-тест, результат не записывается в eval/runs.
    assert calls == ["Когда начало?"]
    assert "reference_answer" not in fake.messages[0][1]["content"]


def test_unmeasured_and_partial_metrics():
    assert aggregate([], [])["v1"]["correct"] is None
    record = {"id": "q01", "version": "v1", "answer": {"sources": []}, "usage": {"prompt_tokens": 200, "completion_tokens": 50}}
    data = aggregate([record], [{"id": "q01", "version": "v1", "correct_manual": "1", "hallucinated_manual": "0"}])
    assert data["v1"]["correct"] == 1
    assert data["v1"]["input_per_question"] == 200
    assert data["v1"]["source_supported"] is None
    assert data["v2"]["has_source"] is None
    assert cost_projection(1_000_000, 500_000, 1000, 1000, 1, 2) == 2
    assert cost_projection(None, 2, 1, 100, 1, 2) is None


def test_export_preserves_manual_labels(tmp_path):
    record = {"id": "q01", "version": "v1", "reference": {"question": "Q", "answerable": False, "reference_answer": "нет", "evidence": None, "source": None},
              "answer": {"answer": REFUSAL, "found": False, "sources": []}, "origin": "gemini", "retrieval": None, "gold_chunks": [], "usage": {"prompt_tokens": 12}}
    save_json(tmp_path / "q01_v1.json", record)
    export_review(tmp_path)
    path = tmp_path / "manual_review.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0]["correct_manual"] = "1"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    assert export_review(tmp_path)[0]["correct_manual"] == "1"


def test_missing_key_is_clear(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match=".env"):
        OpenRouter()


def test_error_analysis_distinguishes_threshold_and_generation():
    record = {"id": "q01", "version": "v2", "question": "Q", "gold_chunks": ["gold:0"], "reference": {"answerable": True},
              "retrieval": {"fragments": [], "candidates_before_threshold": [{"id": "gold:0"}]}}
    labels = [{"id": "q01", "version": "v2", "correct_manual": "0"}]
    assert "порогом" in error_analysis([record], labels)[0]["reason"]
    record["retrieval"]["fragments"] = [{"id": "gold:0"}]
    assert "генерации" in error_analysis([record], labels)[0]["reason"]
