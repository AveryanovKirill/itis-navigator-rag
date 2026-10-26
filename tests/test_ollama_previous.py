import asyncio
import json
import httpx
import pytest

from src.agent import Ollama
from src.previous_practice import validate_previous, PreviousAnswer, tool_context
from src.storage import read_json
from src.config import ROOT
from src.citations import render_single_sentence_sources
from src.schemas import validate_answer


def test_native_ollama_counters_raw_thinking_and_checkpoint(monkeypatch, tmp_path):
    original = httpx.AsyncClient
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.host == "127.0.0.1"
        assert payload["model"] == "qwen3:4b"
        assert payload["stream"] is False and payload["think"] is False
        assert payload["options"]["temperature"] == 0
        assert payload["options"]["num_predict"] == 384
        assert payload["messages"][-1]["content"].endswith("/no_think")
        return httpx.Response(200, json={"model": "qwen3:4b", "message": {"role": "assistant", "content": "{}", "thinking": "TEST_ONLY"},
                                        "prompt_eval_count": 41, "eval_count": 9, "done_reason": "stop"})
    monkeypatch.setattr("src.agent.httpx.AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    llm = Ollama()
    messages = [{"role": "user", "content": "TEST_ONLY"}]
    checkpoint = tmp_path / "call.json"
    record = asyncio.run(llm.complete(messages, checkpoint))
    assert messages[0]["content"] == "TEST_ONLY"  # Не мутируем запрос вызывающего кода.
    assert record["response"]["usage"]["prompt_tokens"] == 41
    assert record["response"]["usage"]["total_tokens"] == 50
    assert record["thinking_returned"] is True
    assert record["raw_provider_response"]["message"]["thinking"] == "TEST_ONLY"
    assert asyncio.run(llm.complete(messages, checkpoint)) == record
    assert len(requests) == 1
    with pytest.raises(ValueError, match="изменился"):
        asyncio.run(llm.complete([{"role": "user", "content": "other"}], checkpoint))


def test_previous_strict_schema_and_actual_tool():
    valid = '{"answer":"Есть ссылка на дневник.","source_tool":"search_docs","found":true}'
    assert validate_previous(valid, "search_docs").found
    with pytest.raises(ValueError, match="не вызывался"):
        validate_previous(valid, "get_group_schedule")
    with pytest.raises(ValueError):
        PreviousAnswer.model_validate_json('{"answer":"text","source_tool":null,"found":"yes"}')
    with pytest.raises(ValueError):
        validate_previous('{"answer":"Точно завтра в 10:00","source_tool":null,"found":true}', None)






def test_citation_display_preserves_claim_and_declared_sources():
    raw = '{"answer":"Заселение 28 августа 2026 года.","sources":[2],"found":true}'
    fragments = [{"number": 2, "text": "Заселение 28 августа 2026 года."}]
    rendered, audit = render_single_sentence_sources(raw, fragments)
    assert audit["original_text"] == json.loads(raw)["answer"]
    assert validate_answer(rendered, fragments).sources == [2]
    assert "[2]" in json.loads(rendered)["answer"]
    bad = raw.replace("28 августа", "29 августа")
    with pytest.raises(ValueError, match="число/дата"):
        validate_answer(render_single_sentence_sources(bad, fragments)[0], fragments)
    invented = raw.replace("[2]", "[3]")
    assert render_single_sentence_sources(invented, fragments) == (invented, None)
    multiple = raw.replace("Заселение 28 августа 2026 года.", "Заселение 28 августа. Потом праздник.")
    assert render_single_sentence_sources(multiple, fragments) == (multiple, None)
