"""Отчёты по настоящим локальным прогонам. Пустые ручные метки не заменяются догадками."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

from .config import ROOT
from .evaluation import run_path, load_results, export_review, aggregate, error_analysis
from .previous_practice import BASE, export as export_previous, LABELS
from .storage import read_json, save_json, now


def show(value):
    return "не измерено" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def write_markdown(path, lines):
    text = "\n".join(lines)
    # Меняется только оформление отчёта; исходные фрагменты остаются в JSON без правок.
    Path(path).write_text("\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n", encoding="utf-8")


def token_stats(records):
    inputs = [r["usage"].get("prompt_tokens") for r in records]
    outputs = [r["usage"].get("completion_tokens") for r in records]
    times = [r["elapsed_seconds"] for r in records]
    calls = [c for r in records for c in r["calls"]]
    return {"completed": len(records), "api_calls": len(calls),
            "input_total": sum(inputs) if inputs and all(x is not None for x in inputs) else None,
            "output_total": sum(outputs) if outputs and all(x is not None for x in outputs) else None,
            "input_mean": statistics.mean(inputs) if inputs and all(x is not None for x in inputs) else None,
            "seconds_total": sum(times) if times else None,
            "seconds_mean": statistics.mean(times) if times else None,
            "seconds_median": statistics.median(times) if times else None,
            "fallback_count": sum(r["origin"] == "validation_fallback" for r in records),
            "format_retries": sum(len(r["calls"]) > 1 for r in records),
            "model_confirmed": all(c["response"].get("model") == "qwen3:4b" for c in calls) if calls else False,
            "thinking_field_nonempty": sum(bool(c.get("thinking_returned")) for c in calls),
            "done_by_length": sum(c.get("raw_provider_response", {}).get("done_reason") == "length" for c in calls)}


def export_rag(name):
    directory = run_path(name)
    records = load_results(directory)
    rows = export_review(directory)
    # Расширенная таблица сохраняет ручные поля оригинального CSV.
    lookup = {(r["id"], r["version"]): r for r in records}
    fields = list(rows[0]) + ["output_tokens", "elapsed_seconds", "api_calls", "raw_final_answer"] if rows else []
    with (directory / "questions_and_answers.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = lookup[(row["id"], row["version"])]
            writer.writerow({**row, "output_tokens": record["usage"]["completion_tokens"],
                             "elapsed_seconds": record["elapsed_seconds"], "api_calls": len(record["calls"]),
                             "raw_final_answer": record["calls"][-1]["response"]["choices"][0]["message"].get("content")})
    questions = read_json(ROOT / "eval/questions_50.json")
    save_json(directory / "all_answers.json", {"run": name, "questions": questions, "answers": records})
    lines = ["# RAG: 50 вопросов, эталоны и реальные ответы", "", f"Запуск {name}; qwen3:4b, температура 0. v1 без документов, v2 с настоящим MCP search_docs. Ответы со статусом validation_fallback — ответ защитного кода, а не успешный ответ модели.", ""]
    for q in questions:
        lines += [f"## {q['id']}. {q['question']}", "", "Эталон: " + q["reference_answer"], "",
                  f"Источник: {q.get('source_url') or 'нет подтверждённого ответа'}; снимок: {q.get('fetched_at') or 'не применимо'}.", ""]
        if q.get("evidence"):
            lines += ["Подтверждение:", "```text", q["evidence"], "```", ""]
        for version in ["v1", "v2"]:
            r = lookup.get((q["id"], version))
            lines += [f"**{version}:** " + (r["answer"]["answer"] if r else "не выполнено"), ""]
            if not r:
                continue
            lines += [f"sources={r['answer']['sources']}; found={r['answer']['found']}; origin={r['origin']}; input={r['usage']['prompt_tokens']}; output={r['usage']['completion_tokens']}; время={r['elapsed_seconds']:.2f} с.", ""]
            if r["retrieval"]:
                lines += ["Переданные источники:"]
                lines += [f"- [{h['number']}] {h['id']}, score={h['score']:.4f}, {h.get('url', '')}" for h in r["retrieval"]["fragments"]]
                lines += [""]
            if r["validation_errors"]:
                lines += ["Ошибки проверки: " + "; ".join(r["validation_errors"]), "", "Сырой последний ответ:", "```json", r["calls"][-1]["response"]["choices"][0]["message"].get("content", ""), "```", ""]
    (directory / "QUESTIONS_AND_ANSWERS_50.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records, rows


def label_metrics(records, rows, versions):
    lookup = {(r["id"], r["version"]): r for r in rows}
    result = {}
    for version in versions:
        subset = [r for r in records if r["version"] == version]
        m = token_stats(subset)
        for field in LABELS:
            values = [lookup.get((r["id"], version), {}).get(field, "") for r in subset]
            if any(v not in {"", "0", "1"} for v in values):
                raise ValueError(f"Поле {field}: допускаются только 0, 1, пусто.")
            m[field] = sum(int(v) for v in values) if values and all(v != "" for v in values) else None
            m[field + "_reviewed"] = sum(v != "" for v in values)
        m["has_source"] = sum(bool(r["answer"].get("source_tool") or r["answer"].get("sources")) for r in subset)
        result[version] = m
    return result


def previous_report(name="ollama_prompts"):
    directory = BASE / "runs" / name
    records, rows = export_previous(directory)
    stats = label_metrics(records, rows, ["v0", "v1"])
    stats30 = label_metrics([r for r in records if r["reference"]["mandatory_30"]], rows, ["v0", "v1"])
    for table in [stats, stats30]:
        for m in table.values():
            m["bad_manual"] = m["completed"] - m["correct_manual"] if m["correct_manual"] is not None else None
    for v in ["v0", "v1"]:
        from .previous_practice import PreviousAnswer
        subset = [r for r in records if r["version"] == v]
        invalid = 0
        for r in subset:
            try:
                PreviousAnswer.model_validate_json(r["calls"][0]["response"]["choices"][0]["message"].get("content", ""))
            except ValueError:
                invalid += 1
        stats[v]["json_schema_errors_first"] = invalid
        stats[v]["validation_failures_first"] = sum(not r["format_first_pass"] for r in subset)
    save_json(BASE / "comparison.json", {"all_50": stats, "mandatory_30": stats30})
    lines = ["# ИТИС Навигатор: промпт v0 → v1", "", "## Цель и условия", "",
             "Предыдущая практическая по промпт-инжинирингу выполнена отдельно от сравнения RAG. В обеих версиях qwen3:4b, температура 0, одинаковая схема JSON, до двух попыток формата и один и тот же сохранённый результат инструмента для каждого вопроса. Изменяется только системный промпт. Ни одна версия не получает эталон.", "",
             "## E1. Классификация ответов", "", "eval/previous_practice/bad_answer_examples.json: 10 авторских примеров с категориями hallucination, off_topic, bad_format, misunderstanding, correct и объяснениями. Это учебные примеры классификации, они не выдаются за реальные ответы модели.", "",
             "## E2. Системный промпт из пяти частей", "", "prompts/previous_v1.md: роль; границы; источники; недостаток информации; формат. v0 — короткая инструкция без подробных ограничений.", "",
             "## E3. Три few-shot примера", "", "В v1 ровно три примера: подтверждённая расшифровка ИТИС, отказ на сказку, недостаток данных о личной оценке. Примеры отделены от текущих источников.", "",
             "## E4. Строгий JSON", "", "PreviousAnswer: answer:str, source_tool:str|null, found:bool, лишние поля запрещены. Проверяется соответствие source_tool реально переданному инструменту, максимум один повтор исправления. При исчерпании — явный validation_fallback. Проверка ссылки не доказывает смысловую правильность.", "",
             "## E5. Реальные прогоны", "", f"Запуск {name}: полный набор 50 вопросов, обязательные первые 30 выделены в questions_30.json. Расписание — сохранённый результат настоящего MCP от 19.10.2026, 02:04 UTC; только три ячейки 11-307/понедельник сверены с официальным PDF. Шаблон не гарантирует актуальные отмены или чётность конкретной недели. Вопросы и эталоны не менялись по ответам модели.", "",
             "Основной файл для просмотра: eval/previous_practice/QUESTIONS_AND_ANSWERS_50.md. Рядом JSON и CSV. Исходные ответы, попытки, фактические счётчики токенов и время — в каталоге запуска.", ""]
    for title, data in [("50 вопросов", stats), ("Обязательные 30 вопросов", stats30)]:
        lines += [f"### {title}", "", "| Метрика | v0 | v1 |", "|---|---|---|"]
        for text, key in [("Реальных завершённых ответов", "completed"), ("Верных (ручные метки)", "correct_manual"), ("Плохих ответов (ручные метки)", "bad_manual"),
                          ("Выдумал факт (ручные метки)", "hallucinated_manual"), ("Непонимание (ручные метки)", "misunderstood_manual"),
                          ("Корректный отказ (ручные метки)", "refusal_correct_manual"), ("Ответ с source_tool", "has_source"),
                          ("Input-токенов всего", "input_total"), ("Output-токенов всего", "output_total"),
                          ("Input на вопрос", "input_mean"), ("Среднее время, с", "seconds_mean"), ("Защитных fallback", "fallback_count")]:
            lines += [f"| {text} | {show(data['v0'][key])} | {show(data['v1'][key])} |"]
        lines += [""]
    lines += [f"Нарушения JSON/типов в первой попытке: v0={stats['v0']['json_schema_errors_first']}, v1={stats['v1']['json_schema_errors_first']}. Нарушения полной проверки (включая source_tool): v0={stats['v0']['validation_failures_first']}, v1={stats['v1']['validation_failures_first']}.", "",
              "Смысловая правильность, галлюцинации и непонимание без заполненных *_manual оставлены «не измерено». Наличие source_tool не означает подтверждения каждого факта. Студент заполняет manual_review.csv (0/1), затем повторяет previous-report. Пустые метки не считаются нулём.", "",
              "## Анализ наблюдаемых ошибок", "", "В реальных ответах встретились смешение бакалавриата 09.03.04 с магистратурой 09.04.04 (p42), неверное направление к регламенту ДОТ за шаблоном практики (p16/v0), подмена вопроса о свободных местах рассказом о личном кабинете (p12). Расширенный промпт не исправляет потерянный retrieval-контекст автоматически. Отказ защитного кода анализируется отдельно от отказа модели.", "",
              "## Расходы и ограничения", "", "Плата за локальный API = 0. Электричество, износ ноутбука и время не нулевые, но не измерены. Показатели Gemini не измерены и эти прогоны не подтверждают её качество. Скрытие thinking не равно отключению: контрольные проверки think:false и /no_think сохранили рассуждение. В рабочих запросах JSON ограничен схемой и 384 выходными токенами."]
    (ROOT / "reports/PREVIOUS_PRACTICE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (BASE / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Метрика (50 вопросов)", "v0", "v1"])
        for key in stats["v0"]:
            writer.writerow([key, show(stats["v0"][key]), show(stats["v1"][key])])
    return stats


def e4_report(name="ollama_e4_citations"):
    records = load_results(run_path(name))
    verification_path = ROOT / "reports/e4_source_verification.json"
    verified = {r["id"]: r for r in read_json(verification_path)["reviews"]} if verification_path.exists() else {}
    lines = ["# E4: реальные ответы через MCP", "", f"Запуск {name}. Локальная qwen3:4b. Пять вопросов с эталонным ответом и два без ответа. Наличие эталона в базе не гарантирует его retrieval или правильную генерацию. Неудачные ответы сохранены.", ""]
    for r in records:
        lines += [f"## {r['id']}. {r['question']}", "", "Вызов: search_docs(query=вопрос), настоящий stdio MCP-клиент и отдельный сервер.", ""]
        for h in r["retrieval"]["fragments"]:
            lines += [f"### [{h['number']}] {h['id']}, score={h['score']:.4f}", "", h.get("url", ""), "", "```text", h["text"], "```", ""]
        lines += ["Ответ системы: " + r["answer"]["answer"], "", "```json", json.dumps(r["answer"], ensure_ascii=False, indent=2), "```", "",
                  "Эталон: " + r["reference"]["reference_answer"], "", f"origin={r['origin']}; input={r['usage']['prompt_tokens']}; output={r['usage']['completion_tokens']}; время={r['elapsed_seconds']:.2f} с.", ""]
        if not r["answer"]["found"] and r["reference"]["answerable"]:
            lines += ["Корректность: неверный/избыточный отказ по эталону. Причину retrieval/генерации проверить по фрагментам выше.", ""]
        elif not r["answer"]["found"]:
            lines += ["Корректность итогового отказа: соответствует отсутствию ответа; fallback не считается успешным ответом модели.", ""]
        else:
            checked = verified.get(r["id"])
            if checked and checked["answer"] == r["answer"] and checked["correct_against_saved_reference"]:
                lines += ["Корректность: первичная проверка Codex по эталону и указанному тексту — совпадает (reports/e4_source_verification.json). Студенту проверить актуальность и подтвердить оценку.", ""]
            else:
                lines += ["Корректность: сверить смысл и источник вручную; формальная валидация пройдена.", ""]
        if r["validation_errors"]:
            lines += ["Ошибки проверки: " + "; ".join(r["validation_errors"]), ""]
    write_markdown(ROOT / "reports/E4_DEMO.md", lines)


def final_report(name="ollama_rag_citations"):
    records, rows = export_rag(name)
    metrics = aggregate(records, rows)
    stats = {v: token_stats([r for r in records if r["version"] == v]) for v in ["v1", "v2"]}
    automatic = {}
    for v in ["v1", "v2"]:
        subset = [r for r in records if r["version"] == v]
        automatic[v] = {"unknown_refused": sum(not r["reference"]["answerable"] and not r["answer"]["found"] for r in subset),
                        "known_refused": sum(r["reference"]["answerable"] and not r["answer"]["found"] for r in subset),
                        "claimed_answer_found": sum(r["answer"]["found"] for r in subset)}
    save_json(ROOT / "reports/automatic_checks.json", {"checks": automatic, "limitation": "found и отказ не доказывают смысловую правильность положительных ответов."})
    save_json(ROOT / "reports/comparison.json", {"manual_metrics": metrics, "observed_usage": stats, "run": name})
    with (ROOT / "reports/comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Метрика (50 вопросов)", "v1", "v2"])
        for title, key in [("Верный ответ (ручные метки)", "correct"), ("Выдумал факт (ручные метки)", "hallucinated"),
                           ("Ответ со ссылкой", "has_source"), ("Input-токенов на вопрос", "input_per_question")]:
            writer.writerow([title, show(metrics["v1"][key]), show(metrics["v2"][key])])
    initial = (ROOT / "reports/before_ollama/FINAL_REPORT.md").read_text(encoding="utf-8")
    preserved = initial[initial.index("## E1:"):initial.index("## E4:")]
    tests = read_json(ROOT / "reports/test_results.json")
    lines = ["# ИТИС Навигатор — итоговый отчёт RAG E1–E5", "", "## Цель работы", "", "Создать проверяемый RAG-помощник по официальным документам ИТИС КФУ и сравнить обычный запрос с поиском через настоящий MCP.", "",
             "## Технологии", "", "Python 3.12, NumPy, multilingual-e5-small, официальный MCP SDK, Pydantic, httpx, pytest. Генерация: уже установленная локальная Ollama 0.40.6 / qwen3:4b Q4_K_M, температура 0. OpenRouter/Gemini-клиент сохранён как дополнительный провайдер, но реальные результаты здесь относятся только к Qwen. Причина замены — срочное уточнение пользователя и отсутствие API-ключа.", "",
             "## Архитектура", "", "v1: вопрос → Qwen → проверка JSON. v2: вопрос → stdio MCP search_docs → сохранённый NumPy-индекс → до трёх фрагментов выше калиброванного порога → Qwen → Pydantic/проверка ссылок → ответ. Эталоны не передаются модели. Расписание обслуживает отдельный get_group_schedule, оно не входит в RAG E5.", "", preserved,
             "## E4: реальный MCP и генерация", "", "Выполнена сквозная цепочка вопрос → настоящий MCP → Qwen → строгий Answer(answer,sources,found). Семь контрольных вопросов, сырые ответы и источники — reports/E4_DEMO.md и eval/runs/ollama_e4_citations/. Среди известных вопросов возможны избыточные отказы; они не объявляются правильными. Два неизвестных вопроса проверены тем же механизмом. Фиктивные ссылки/числа и нестандартные отказы вызывают повтор, затем явно помеченный fallback.", "",
             "Если модель выбрала существующие sources, но не вставила [n] в одно короткое предложение, приложение отображает эти уже выбранные номера перед точкой. Текст факта и список sources не меняются, событие записывается в answer_transformations. Для нескольких предложений автоматическая расстановка не применяется. Затем выполняются те же строгие проверки ссылок и чисел. Исходный ответ модели всегда сохранён в calls/raw_provider_response. Это оформление ссылки, не доказательство поддержки утверждения источником.", "",
             "Формальная валидность JSON и существование источника не доказывают смысловую правильность. Проверка источников и утверждений остаётся ручной.", "",
             "## E5: сравнение 50×2", "", f"Основной локальный запуск: {name}. В обеих версиях одна Qwen3:4b, температура 0, одинаковые 50 вопросов (40 с ответом и 10 без). Результаты каждого реального ответа, повторов и usage сохранены немедленно. v1 не получает документы и не вызывает поиск.", "",
             "| Метрика (50 вопросов) | v1 | v2 |", "|---|---|---|"]
    for title, key in [("Завершённых ответов", "completed"), ("Верный ответ (ручные метки)", "correct"),
                       ("Выдумал факт (ручные метки)", "hallucinated"), ("Ответ со ссылкой", "has_source"),
                       ("Источник подтверждает ответ (ручные метки)", "source_supported"), ("Корректных отказов (ручные метки)", "correct_refusal"),
                       ("Input-токенов на вопрос", "input_per_question")]:
        lines += [f"| {title} | {show(metrics['v1'][key])} | {show(metrics['v2'][key])} |"]
    for title, key in [("Input-токенов всего", "input_total"), ("Output-токенов всего", "output_total"),
                       ("Среднее время ответа, с", "seconds_mean"), ("Медиана времени, с", "seconds_median"),
                       ("Ответов защитного fallback", "fallback_count"), ("Повторов формата", "format_retries"), ("API-вызовов с лимитом длины", "done_by_length")]:
        lines += [f"| {title} | {show(stats['v1'][key])} | {show(stats['v2'][key])} |"]
    for title, key in [("Отказов на 10 вопросов без эталонного ответа (проверка статуса)", "unknown_refused"),
                       ("Отказов на 40 вопросов с эталонным ответом", "known_refused")]:
        lines += [f"| {title} | {automatic['v1'][key]} | {automatic['v2'][key]} |"]
    ratio = stats["v2"]["input_mean"] / stats["v1"]["input_mean"] if stats["v1"]["input_mean"] and stats["v2"]["input_mean"] is not None else None
    lines += ["", f"Отношение средних входных токенов v2/v1: {show(ratio)}. Время в таблице — обработка LLM и валидация, без старта MCP и загрузки эмбеддингов; API-счётчики учитывают также кэшированные токены запроса."]
    lines += ["", "Полные вопросы, эталоны и оба ответа: eval/runs/" + name + "/QUESTIONS_AND_ANSWERS_50.md, all_answers.json, questions_and_answers.csv. Для ручной разметки — manual_review.csv, поля *_manual: 1/0, пусто не проверено. Неизвестные метрики не заменены красивыми числами.", "",
              "## Анализ ошибок и изменения промпта", "", "Первый полный запуск ollama_rag с длинным v2-промптом сохранён отдельно: избыточные отказы, 4 фактических ответа со ссылками из 50 v2. Во втором ollama_rag_short модель копировала шаблон «Краткий подтверждённый факт». Шаблон удалён; в ollama_rag_final модель чаще извлекала реальные факты, но не вставляла текстовые маркеры при заполненных sources. Добавлено явное отображение уже выбранных ссылок для одного предложения; последний запуск — ollama_rag_citations. Все исходные прогоны и старые промпты сохранены. Корпус, векторы, порог и эталоны не менялись. Результаты вручную не исправлялись.", "",
              "Это подбор промпта на разработческом наборе, а не независимая оценка на невидимых вопросах. Улучшение нельзя объявлять универсальным. Ошибки итогового v2: при отсутствии эталона в сыром top-3 — retrieval; при наличии до порога — фильтрация; при наличии в переданных фрагментах — генерация/промпт. Детали — reports/ERROR_ANALYSIS.md. Предварительные наблюдения конкретных смысловых ошибок — reports/OBSERVED_ERRORS.md (например, неверная почта itus вместо itis и перевёрнутая причинная связь практики и стипендии). Неверный источник может существовать и всё равно не подтверждать ответ.", "",
              "## Цена использования LLM", "", "Локальный Ollama API: денежная плата провайдеру 0. Счётчики prompt_eval_count и eval_count взяты из реальных ответов native API; повторы включены. Это не токенизация Gemini/OpenRouter и не денежный тариф. Издержки электричества, износа и времени не измерены. Полный учёт всех сохранённых опытов, отладки и демонстраций — reports/TOKEN_BUDGET.md и TOKEN_BUDGET.json; расходы разработки отделены от итогового сравнения.", ""]
    for v in ["v1", "v2"]:
        m = stats[v]
        lines += [f"- {v}: input={show(m['input_total'])}, output={show(m['output_total'])}; API-вызовов={m['api_calls']}; суммарное время генеративных обработок={show(m['seconds_total'])} с."]
    lines += ["", "Прогноз токенов для N вопросов: N × среднее число входных/выходных токенов по данной версии. Облачная формула при переходе к OpenRouter: (input × INPUT_PRICE_PER_MILLION + output × OUTPUT_PRICE_PER_MILLION) / 1 000 000. Тарифы не заданы, бюджет ДЗ 2 не предоставлен. Их нельзя восстановить из локального прогона.", "",
              "## GPU, thinking и время", "", "Один экземпляр qwen3:4b, все запросы последовательны. /api/ps показал 3 877 571 133 байта VRAM и контекст 8192. Проверки: reports/ollama_probe.json, ollama_probe_completion.json, ollama_soft_switch_probe.json. think:false и /no_think в данной установленной конфигурации не доказали отключения рассуждений: контрольная арифметика продолжила рассуждение. При think:false оно попадало в content, при think:true — в thinking. Поэтому отключение не заявляется. JSON-схема и предел 384 токена ограничивают рабочий вывод, но не доказывают отключение внутреннего рассуждения.", "",
              "## Ограничения и проверка", "", f"pytest: {tests['passed']} passed, errors={tests['errors']}, failures={tests['failures']}, skipped={tests['skipped']}. Журнал — reports/pytest_output.txt. Эмбеддинги и production stdio MCP проверяются настоящими интеграционными тестами. Подмены API используются только в unit-тестах, в эксперимент они не включены.", "",
              "Веб-снимки официальных страниц получены через веб-сервис из-за прямых тайм-аутов; актуальность каждого правила на день сдачи требует проверки. Регламент ДОТ опубликован в 2020 году, отмена отдельно не проверена. Google-таблица расписания сверена с официальным PDF только для трёх ячеек. Оценка смыслового качества и галлюцинаций требует разметки студентом. Реальные Gemini E4/E5 не выполнены; локальные Qwen-прогоны их не подтверждают.", "",
              "## Вывод", "", "E1–E3 сохранены с реальными локальными измерениями, MCP и оба последовательных генеративных эксперимента выполнены локально. Работа показывает не только источники, но и реальные ограничения retrieval, промпта и небольшой модели. Утверждать уменьшение галлюцинаций следует после проверки всех ответов по источникам."]
    (ROOT / "reports/FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    errors = error_analysis(records, rows)
    # Дополнительно точные наблюдаемые избыточные отказы, без семантического судьи.
    obvious = [r for r in records if r["version"] == "v2" and r["reference"]["answerable"] and not r["answer"]["found"]]
    details = []
    for r in obvious:
        gold = set(r["gold_chunks"])
        hit = bool(gold & {h["id"] for h in r["retrieval"]["fragments"]})
        raw = bool(gold & {h["id"] for h in r["retrieval"].get("candidates_before_threshold", [])})
        details.append({"id": r["id"], "question": r["question"], "kind": "избыточный отказ по эталону",
                        "gold_in_context": hit, "gold_in_raw_top3": raw, "origin": r["origin"],
                        "likely_stage": "generation/validation" if hit else "threshold" if raw else "retrieval"})
    save_json(ROOT / "reports/error_analysis.json", {"manual_errors": errors, "observable_false_refusals": details})
    (ROOT / "reports/ERROR_ANALYSIS.md").write_text("# Анализ ошибок v2\n\nСемантические ошибки по ручной разметке: " + str(len(errors)) + ". Пустая разметка не доказывает отсутствие ошибок. Конкретные предварительные наблюдения — OBSERVED_ERRORS.md; они также занесены в notes CSV.\n\nНаблюдаемые избыточные отказы на вопросы с эталонным ответом:\n\n" + "\n".join(f"- {r['id']}: {r['likely_stage']}; эталон в контексте={r['gold_in_context']}; в сыром top-3={r['gold_in_raw_top3']}; origin={r['origin']}" for r in details) + "\n", encoding="utf-8")
    e4_report()
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="ollama_rag_citations")
    parser.add_argument("--previous-run", default="ollama_prompts")
    args = parser.parse_args()
    final_report(args.run)
    previous_report(args.previous_run)
    print("Отчёты обновлены по сохранённым реальным ответам; ручные метки сохранены.")
