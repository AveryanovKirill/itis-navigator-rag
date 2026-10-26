"""Проверки сетевого протокола с httpx.MockTransport; настоящие API-ответы не подменяются в отчётах."""
import asyncio
import httpx
import pytest
from src.agent import OpenRouter, APIError
from src.storage import read_json


def install_transport(monkeypatch, handler):
    import src.agent
    original = httpx.AsyncClient
    monkeypatch.setattr(src.agent.httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    async def immediate(delay):
        pass
    monkeypatch.setattr(src.agent.asyncio, "sleep", immediate)


def test_temporary_failure_usage_checkpoint_and_resume(monkeypatch, tmp_path):
    calls = []
    def handler(request):
        import json
        payload = json.loads(request.content)
        assert payload["temperature"] == 0
        assert payload["model"] == "google/gemini-2.5-flash"
        assert payload["response_format"]["json_schema"]["strict"] is True
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"code": 429}})
        return httpx.Response(200, json={"model": payload["model"], "choices": [{"message": {"content": "{}"}}],
                                        "usage": {"prompt_tokens": 23, "completion_tokens": 3}})
    install_transport(monkeypatch, handler)
    llm = OpenRouter(key="TEST_ONLY")
    path = tmp_path / "call.json"
    first = asyncio.run(llm.complete([{"role": "user", "content": "Q"}], checkpoint=path))
    assert first["http_attempts"] == 2
    assert read_json(path)["response"]["usage"]["prompt_tokens"] == 23
    assert asyncio.run(llm.complete([{"role": "user", "content": "Q"}], checkpoint=path)) == first
    assert len(calls) == 2
    with pytest.raises(ValueError, match="изменились"):
        asyncio.run(llm.complete([{"role": "user", "content": "Other"}], checkpoint=path))


def test_permanent_api_error_is_not_success(monkeypatch):
    install_transport(monkeypatch, lambda r: httpx.Response(401, json={"error": {"code": 401}}))
    with pytest.raises(APIError, match="OPENROUTER_API_KEY"):
        asyncio.run(OpenRouter(key="TEST_ONLY").complete([]))


def test_limited_retries(monkeypatch):
    calls = []
    def handler(r):
        calls.append(1)
        return httpx.Response(503, json={"error": {"code": 503}})
    install_transport(monkeypatch, handler)
    with pytest.raises(APIError):
        asyncio.run(OpenRouter(key="TEST_ONLY", retries=3).complete([]))
    assert len(calls) == 3
