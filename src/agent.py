import asyncio
import json
import time
from pathlib import Path
import httpx

from .config import ROOT, MODEL, LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SECONDS, api_key
from .mcp_client import call_search
from .schemas import Answer, validate_answer, fallback
from .storage import save_json, read_json, fingerprint, now


class APIError(RuntimeError):
    pass


class OpenRouter:
    """Реальный API. Подмены используются только в tests/."""
    def __init__(self, key=None, model=MODEL, retries=3):
        self.backend = "openrouter"
        self.key = key or api_key()
        self.model = model
        self.retries = retries

    async def complete(self, messages, checkpoint=None):
        payload = {"model": self.model, "temperature": 0, "max_tokens": 1800,
                   "messages": messages, "provider": {"require_parameters": True},
                   "response_format": {"type": "json_schema", "json_schema": {
                       "name": "Answer", "strict": True, "schema": Answer.model_json_schema()}}}
        request_hash = fingerprint(payload)
        if checkpoint and Path(checkpoint).exists():
            cached = read_json(checkpoint)
            if cached["request_hash"] != request_hash:
                raise ValueError("Настройки запроса изменились; используйте новый каталог эксперимента.")
            return cached
        async with httpx.AsyncClient(timeout=90) as client:
            for attempt in range(self.retries):
                try:
                    response = await client.post("https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.key}", "X-Title": "University RAG Practice"}, json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt + 1 == self.retries:
                        raise APIError("Не удалось связаться с OpenRouter после ограниченного числа попыток. Результаты уже завершённых запросов сохранены.") from exc
                    await asyncio.sleep(2 ** attempt)
                    continue
                if response.status_code in {408, 429, 500, 502, 503, 504} and attempt + 1 < self.retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if response.status_code != 200:
                    friendly = {401: "Проверьте OPENROUTER_API_KEY.", 402: "Недостаточно баланса OpenRouter.",
                                403: "Доступ ограничен провайдером.", 429: "Лимит запросов. Повторите запуск позже.",
                                400: "Провайдер отклонил схему или параметры. Проверьте поддержку structured outputs."}
                    raise APIError(f"OpenRouter HTTP {response.status_code}. " + friendly.get(response.status_code, "Повторите запуск позже."))
                try:
                    data = response.json()
                except ValueError as exc:
                    raise APIError("OpenRouter вернул не JSON; повторите запуск.") from exc
                # Сохраняем каждый полученный 200-ответ до разбора формата модели.
                record = {"request_hash": request_hash, "requested_model": self.model, "temperature": 0,
                          "timestamp": now(), "http_attempts": attempt + 1, "response": data}
                if checkpoint:
                    save_json(checkpoint, record)
                return record
        raise APIError("Не удалось получить ответ OpenRouter.")


class Ollama:
    """Только уже установленная локальная Qwen. Ни pull, ни облачных вызовов."""
    def __init__(self, schema=None, retries=3):
        self.model = OLLAMA_MODEL
        self.backend = "ollama"
        self.retries = retries
        self.schema = schema or Answer.model_json_schema()
        self.options = {"temperature": 0, "num_ctx": 8192, "num_predict": 384, "seed": 42}

    async def complete(self, messages, checkpoint=None):
        # Soft switch документирован Qwen; одинаков для всех сравниваемых версий.
        messages = [dict(m) for m in messages]
        messages[-1]["content"] += "\n/no_think"
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "think": False, "format": self.schema, "options": self.options,
                   "keep_alive": "10m"}
        request_hash = fingerprint(payload)
        if checkpoint and Path(checkpoint).exists():
            cached = read_json(checkpoint)
            if cached["request_hash"] != request_hash:
                raise ValueError("Запрос изменился. Используйте новый каталог эксперимента.")
            return cached
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS, trust_env=False) as client:
            for attempt in range(self.retries):
                start = time.perf_counter()
                try:
                    response = await client.post(OLLAMA_BASE_URL + "/api/chat", json=payload)
                    if response.status_code in {408, 429, 500, 502, 503, 504} and attempt + 1 < self.retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    if response.status_code != 200:
                        raise APIError(f"Ollama HTTP {response.status_code}: {response.text[:300]}. Убедитесь, что qwen3:4b уже установлена.")
                    raw = response.json()
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt + 1 == self.retries:
                        raise APIError(f"Нет доступа к Ollama {OLLAMA_BASE_URL}. Запустите Ollama и выполните ollama pull qwen3:4b, если модель ещё не установлена. Проверьте python -m src.cli doctor. Модели автоматически не скачиваются.") from exc
                    await asyncio.sleep(2 ** attempt)
                    continue
                elapsed = time.perf_counter() - start
                prompt, completion = raw.get("prompt_eval_count"), raw.get("eval_count")
                usage = {"prompt_tokens": prompt, "completion_tokens": completion,
                         "total_tokens": prompt + completion if isinstance(prompt, int) and isinstance(completion, int) else None,
                         "cost": 0}
                record = {"request_hash": request_hash, "requested_model": self.model, "temperature": 0,
                          "timestamp": now(), "http_attempts": attempt + 1, "elapsed_seconds": elapsed,
                          "provider": "ollama", "think_requested": False,
                          "soft_no_think": True,
                          "thinking_returned": bool(raw.get("message", {}).get("thinking")),
                          "raw_provider_response": raw,
                          "response": {"model": raw.get("model"), "choices": [{"message": raw.get("message", {})}], "usage": usage}}
                if checkpoint:
                    save_json(checkpoint, record)
                return record
        raise APIError("Не удалось получить ответ Ollama.")


def create_llm():
    if LLM_PROVIDER == "ollama":
        return Ollama()
    if LLM_PROVIDER == "openrouter":
        return OpenRouter()
    raise ValueError("LLM_PROVIDER должен быть ollama или openrouter.")


async def ask(question, version, llm, mcp=None, checkpoint_dir=None, retrieval=None):
    started = time.perf_counter()
    if version not in {"v1", "v2"}:
        raise ValueError("Допустимы только v1 и v2.")
    if version == "v1":
        retrieved = None  # Здесь нет вызова поиска и нет документов в prompt.
        user = json.dumps({"question": question}, ensure_ascii=False)
    else:
        if mcp is None:
            raise ValueError("Для v2 нужен подключённый MCP-клиент.")
        retrieved = retrieval or await call_search(mcp, question)
        user = json.dumps({"question": question, "search_docs": retrieved}, ensure_ascii=False)
    fragments = retrieved["fragments"] if retrieved else []
    messages = [{"role": "system", "content": (ROOT / f"prompts/system_{version}.md").read_text(encoding="utf-8")},
                {"role": "user", "content": user}]
    calls, errors, transformations = [], [], []
    validated = None
    for attempt in range(2):
        checkpoint = Path(checkpoint_dir) / f"completion_{attempt}.json" if checkpoint_dir else None
        response_record = await llm.complete(messages, checkpoint=checkpoint)
        calls.append(response_record)
        data = response_record["response"]
        raw = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        try:
            if data.get("error") or not isinstance(raw, str):
                raise ValueError("API не вернул текст ответа.")
            candidate = raw
            if version == "v2":
                from .citations import render_single_sentence_sources
                candidate, transformation = render_single_sentence_sources(raw, fragments)
                if transformation:
                    transformations.append({"attempt": attempt, **transformation})
            validated = validate_answer(candidate, fragments, version)
            break
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
            if attempt == 0:
                messages += [{"role": "assistant", "content": raw or ""},
                             {"role": "user", "content": "Исправь формат и ссылки. " + str(exc) + " Соблюдай исходный системный промпт, верни только JSON."}]
    result = {"version": version, "question": question, "answer": (validated or fallback()).model_dump(),
              "origin": ("ollama" if getattr(llm, "backend", "openrouter") == "ollama" else "gemini") if validated else "validation_fallback", "backend": getattr(llm, "backend", "openrouter"),
              "retrieval": retrieved, "validation_errors": errors, "calls": calls, "timestamp": now(),
              "answer_transformations": transformations,
              "semantic_status": "Требуется ручная проверка смысла и подтверждения каждого утверждения."}
    usages = [c["response"].get("usage", {}) for c in calls]
    result["usage"] = {name: sum(u[name] for u in usages) if usages and all(isinstance(u.get(name), (int, float)) for u in usages) else None
                       for name in ["prompt_tokens", "completion_tokens", "total_tokens", "cost"]}
    result["usage"]["complete"] = all(isinstance(u.get("prompt_tokens"), int) and isinstance(u.get("completion_tokens"), int) for u in usages)
    result["resolved_models"] = [c["response"].get("model") for c in calls]
    result["model_confirmed"] = all(m == llm.model for m in result["resolved_models"]) if hasattr(llm, "model") else False
    result["elapsed_seconds"] = time.perf_counter() - started
    result["generation_seconds"] = sum(c.get("elapsed_seconds", 0) for c in calls)
    return result
