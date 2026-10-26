"""Предыдущая практика: два промпта при одинаковых инструментах и данных."""
import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

from .agent import Ollama, APIError
from .config import ROOT, LLM_PROVIDER
from .mcp_client import connect_mcp, call_search
from .storage import read_json, save_json, fingerprint, now

BASE = ROOT / "eval/previous_practice"
LABELS = ["correct_manual", "hallucinated_manual", "misunderstood_manual", "refusal_correct_manual", "source_support_manual"]


class PreviousAnswer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    answer: str = Field(min_length=1)
    source_tool: str | None
    found: bool


def validate_previous(raw, tool):
    value = PreviousAnswer.model_validate_json(raw)
    if value.source_tool is not None and value.source_tool != tool:
        raise ValueError("Указан инструмент, который не вызывался для этого вопроса.")
    if value.found and (not tool or value.source_tool != tool):
        raise ValueError("Для фактического ответа укажите реально переданный инструмент.")
    if not value.found and value.source_tool is not None:
        raise ValueError("При недостатке данных source_tool должен быть null.")
    return value


async def tool_context(q, session):
    if q["kind"] == "docs" or q["kind"] == "unknown":
        result = await call_search(session, q["question"])
        return {"tool": "search_docs", "transport": "real stdio MCP", "result": result}
    if q["kind"] == "schedule":
        # Один и тот же датированный реальный MCP-снимок для обеих версий, без смешения дат.
        snapshot = read_json(BASE / "schedule_snapshot.json")["data"]
        return {"tool": "get_group_schedule", "transport": "frozen real MCP result",
                "snapshot_file": "eval/previous_practice/schedule_snapshot.json",
                "result": {key: snapshot[key] for key in ["group", "day", "lessons", "fetchedAt", "url", "official_pdf_year4", "verified_against_official_pdf", "provenance_note"]}}
    return {"tool": None, "result": None,
            "routing_note": "Для неоднозначного вопроса нужны уточнения; постороннему вопросу университетские данные не соответствуют."}


async def generate(q, version, context, llm, directory):
    # Эталоны и поле answerable НЕ передаются модели.
    messages = [{"role": "system", "content": (ROOT / f"prompts/previous_{version}.md").read_text(encoding="utf-8")},
                {"role": "user", "content": json.dumps({"question": q["question"], "tool_result": context}, ensure_ascii=False)}]
    errors, calls, parsed = [], [], None
    started = time.perf_counter()
    for attempt in range(2):
        call = await llm.complete(messages, checkpoint=directory / f"completion_{attempt}.json")
        calls.append(call)
        raw = call["response"]["choices"][0]["message"].get("content", "")
        try:
            parsed = validate_previous(raw, context["tool"])
            break
        except ValueError as exc:
            errors.append(str(exc))
            messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": "Исправь JSON и source_tool: " + str(exc)}]
    answer = parsed or PreviousAnswer(answer="Недостаточно проверенной информации для ответа.", source_tool=None, found=False)
    usage = {key: sum(c["response"]["usage"][key] for c in calls) if all(c["response"]["usage"].get(key) is not None for c in calls) else None
             for key in ["prompt_tokens", "completion_tokens", "total_tokens", "cost"]}
    return {"id": q["id"], "version": version, "question": q["question"], "reference": q,
            "answer": answer.model_dump(), "origin": "ollama" if parsed else "validation_fallback",
            "raw_final_answer": calls[-1]["response"]["choices"][0]["message"].get("content"),
            "calls": calls, "tool_context": context, "context_hash": fingerprint(context), "usage": usage,
            "validation_errors": errors, "format_first_pass": not errors, "format_final": parsed is not None,
            "elapsed_seconds": time.perf_counter() - started, "timestamp": now()}


def load_previous(directory):
    return [read_json(p) for p in sorted(Path(directory).glob("p*_v[01].json"))]


async def experiment(name="ollama_prompts", limit=50):
    if LLM_PROVIDER != "ollama":
        raise ValueError("Срочный локальный эксперимент выполняется только с LLM_PROVIDER=ollama.")
    if not 1 <= limit <= 50 or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("limit: 1–50; имя запуска: латинские буквы, цифры, _ и -.")
    directory = BASE / "runs" / name
    questions = read_json(BASE / "questions_50.json")
    llm = Ollama(schema=PreviousAnswer.model_json_schema())
    settings = {"model": llm.model, "provider": "ollama", "options": llm.options, "temperature": 0,
                "dataset_hash": fingerprint(questions), "index_hash": fingerprint(read_json(ROOT / "index/metadata.json")),
                "schedule_hash": fingerprint(read_json(BASE / "schedule_snapshot.json")),
                "prompts": {v: fingerprint((ROOT / f"prompts/previous_{v}.md").read_text(encoding="utf-8")) for v in ["v0", "v1"]},
                "implementation_hash": fingerprint({p: (ROOT / "src" / p).read_text(encoding="utf-8") for p in ["agent.py", "previous_practice.py"]}),
                "format_attempts": 2, "think": False, "soft_no_think": True,
                "procedure": "identical cached real tool result per question for v0/v1; same JSON schema and repair; schedule fixed snapshot"}
    manifest = directory / "manifest.json"
    if manifest.exists() and read_json(manifest)["settings"] != settings:
        recorded = read_json(manifest)["settings"]
        same_protocol = {k: v for k, v in recorded.items() if k != "implementation_hash"} == {k: v for k, v in settings.items() if k != "implementation_hash"}
        complete = all((directory / f"{q['id']}_{v}.json").exists() for q in questions[:limit] for v in ["v0", "v1"])
        if same_protocol and complete:
            # Уже готовые ответы читаем по исходному manifest; новых вызовов изменённым кодом нет.
            export(directory)
            print("Все выбранные ответы уже сохранены. Экспорт обновлён; исходный manifest сохранён, новых LLM-вызовов нет.")
            return directory
        raise ValueError("Настройки изменились. Для новых вызовов используйте новое имя --run.")
    if not manifest.exists():
        save_json(manifest, {"created_at": now(), "settings": settings})
    async with connect_mcp() as session:
        for q in questions[:limit]:
            path = directory / "contexts" / f"{q['id']}.json"
            context = read_json(path) if path.exists() else await tool_context(q, session)
            save_json(path, context)
            for version in ["v0", "v1"]:
                target = directory / f"{q['id']}_{version}.json"
                if target.exists():
                    print(f"{q['id']} {version}: уже сохранён, пропуск.", flush=True)
                    continue
                try:
                    result = await generate(q, version, context, llm, directory / "calls" / f"{q['id']}_{version}")
                except APIError as exc:
                    save_json(directory / "last_error.json", {"id": q["id"], "version": version, "error": str(exc), "timestamp": now()})
                    raise
                save_json(target, result)
                print(f"{q['id']} {version}: {result['origin']}; found={result['answer']['found']}; input={result['usage']['prompt_tokens']}; секунд={result['elapsed_seconds']:.2f}", flush=True)
            if q["id"] in {"p05", "p30", "p50"}:
                export(directory)
    export(directory)
    return directory


def export(directory):
    directory = Path(directory)
    records = load_previous(directory)
    path = directory / "manual_review.csv"
    old = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            old = {(r["id"], r["version"]): r for r in csv.DictReader(f)}
    fields = ["id", "version", "mandatory_30", "kind", "question", "reference_answer", "evidence", "source_url", "fetched_at", "answer", "raw_final_answer", "found", "source_tool", "origin", "format_first_pass", "format_final", "input_tokens", "output_tokens", "elapsed_seconds", *LABELS, "notes"]
    rows = []
    for r in records:
        q = r["reference"]
        row = {"id": r["id"], "version": r["version"], **{k: q.get(k) for k in ["mandatory_30", "kind", "question", "reference_answer", "evidence", "source_url", "fetched_at"]},
               **{k: r[k] for k in ["origin", "raw_final_answer", "format_first_pass", "format_final", "elapsed_seconds"]},
               **r["answer"], "input_tokens": r["usage"]["prompt_tokens"], "output_tokens": r["usage"]["completion_tokens"]}
        prior = old.get((r["id"], r["version"]), {})
        for label in [*LABELS, "notes"]:
            row[label] = prior.get(label, "")
        rows.append(row)
    directory.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save_json(directory / "all_answers.json", records)
    save_json(BASE / "QUESTIONS_AND_ANSWERS_50.json", {"questions": read_json(BASE / "questions_50.json"), "answers": records, "run": directory.name})
    (BASE / "QUESTIONS_AND_ANSWERS_50.csv").write_bytes(path.read_bytes())
    lookup = {(r["id"], r["version"]): r for r in records}
    lines = ["# ИТИС Навигатор: промпт v0 → v1 — 50 вопросов", "", f"Запуск: {directory.name}. Реальная локальная qwen3:4b, температура 0. Первые 30 — обязательная часть, 31–50 — расширение.", "", "Обе версии получают один и тот же результат инструмента. Расписание — фиксированный датированный снимок, без обещания текущих отмен. Семантические метки требуют ручной проверки.", ""]
    for q in read_json(BASE / "questions_50.json"):
        lines += [f"## {q['id']}. {q['question']}", "", "Эталон / ожидаемое поведение: " + q["reference_answer"], "",
                  f"Источник: {q.get('source_url') or 'Ответ отсутствует / требуется уточнение / вне области'}. Дата получения: {q.get('fetched_at') or 'не применимо'}.", ""]
        if q.get("evidence"):
            lines += ["Подтверждение:", "", "```text", q["evidence"], "```", ""]
        for version in ["v0", "v1"]:
            r = lookup.get((q["id"], version))
            lines += [f"**{version}:** " + (r["answer"]["answer"] if r else "не выполнено"), ""]
            if r:
                lines += [f"source_tool={r['answer']['source_tool']}; found={r['answer']['found']}; origin={r['origin']}; input={r['usage']['prompt_tokens']}; output={r['usage']['completion_tokens']}; {r['elapsed_seconds']:.2f} с.", ""]
                if r["validation_errors"]:
                    lines += ["Ошибки проверки: " + "; ".join(r["validation_errors"]), "", "Сырой последний ответ:", "```json", r["raw_final_answer"], "```", ""]
    (BASE / "QUESTIONS_AND_ANSWERS_50.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records, rows


def main():
    parser = argparse.ArgumentParser(description="Предыдущая практика: сравнение промптов при одинаковых инструментах")
    parser.add_argument("--run", default="ollama_prompts")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.export_only:
            export(BASE / "runs" / args.run)
        else:
            asyncio.run(experiment(args.run, args.limit))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
