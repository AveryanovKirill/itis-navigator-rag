import csv
import os
from .config import ROOT, EMBEDDING_MODEL
from .evaluation import run_path, export_review, load_results, aggregate, error_analysis, REVIEW_FIELDS
from .storage import read_json, save_json


def optional(path, default=None):
    return read_json(ROOT / path) if (ROOT / path).exists() else default


def measured(value, suffix=""):
    return "не измерено" if value is None else f"{value:.2f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"


def cost_projection(input_total, output_total, observed_questions, target_questions, input_price=None, output_price=None):
    if any(x is None for x in [input_total, output_total, input_price, output_price]) or not observed_questions:
        return None
    if min(input_price, output_price, target_questions) < 0:
        raise ValueError("Цены и число вопросов не могут быть отрицательными.")
    return (input_total * input_price + output_total * output_price) / 1_000_000 * target_questions / observed_questions


def create_review_template():
    questions = optional("eval/questions_50.json", [])
    path = ROOT / "reports/manual_review_template.csv"
    if path.exists():
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for q in questions:
            for version in ["v1", "v2"]:
                writer.writerow({"id": q["id"], "version": version, "question": q["question"], "answerable": q["answerable"],
                                 "reference_answer": q["reference_answer"], "evidence": q["evidence"], "reference_source": q["source"],
                                 "notes": "Ожидает реального API-прогона; метки пока не выставлять."})


def generate_report(name="main", target_questions=1000):
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    directory = run_path(name)
    results = load_results(directory)
    reviews = export_review(directory)
    metrics = aggregate(results, reviews)
    save_json(reports / "comparison.json", metrics)
    with (reports / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Метрика (по выполненным вопросам)", "v1", "v2"])
        for title, field in [("Завершённых вопросов", "completed"), ("Верный ответ по ручному эталону", "correct"),
                             ("Выдумал факт (ручная оценка)", "hallucinated"), ("Ответ со ссылкой", "has_source"),
                             ("Источник подтверждает ответ (ручная оценка)", "source_supported"),
                             ("Input-токенов на вопрос", "input_per_question"), ("Корректный отказ", "correct_refusal")]:
            writer.writerow([title] + [measured(metrics[v][field]) for v in ["v1", "v2"]])
    errors = error_analysis(results, reviews)
    save_json(reports / "error_analysis.json", errors)
    e1, chunks = optional("reports/e1_results.json"), optional("index/chunks.json", [])
    e2, e3, threshold = optional("reports/e2_results.json"), optional("reports/recall.json"), optional("reports/threshold.json")
    mcp_local = optional("reports/mcp_local_demo.json")
    tests = optional("reports/test_results.json")
    price_in = float(os.environ["INPUT_PRICE_PER_MILLION"]) if os.getenv("INPUT_PRICE_PER_MILLION", "").strip() else None
    price_out = float(os.environ["OUTPUT_PRICE_PER_MILLION"]) if os.getenv("OUTPUT_PRICE_PER_MILLION", "").strip() else None
    rows = []
    for label, field in [("Верный ответ по эталону", "correct"), ("Выдумал факт", "hallucinated"),
                          ("Ответ со ссылкой на источник", "has_source"), ("Input-токенов на вопрос", "input_per_question")]:
        rows.append("| " + " | ".join([label] + [measured(metrics[v][field]) for v in ["v1", "v2"]]) + " |")
    report = ["# Итоговый отчёт по учебной RAG-системе", "", "## Цель работы", "",
              "Реализовать E1–E5: смысловые векторы, нарезку, векторный поиск, MCP, ответы Gemini с источниками и честное сравнение двух версий.", "",
              "## Использованные технологии", "", f"Python 3.11+ (проверено на 3.12.14), {EMBEDDING_MODEL} (CPU), NumPy, официальный MCP SDK 1.x, Pydantic 2, httpx, pytest, dotenv. Точные установленные версии — requirements-lock.txt.", "",
              "Корпус: 5 официальных текстовых снимков ИТИС/КФУ. Прямой HTTPS к КФУ завершался тайм-аутом; тексты получены через веб-сервис, исходные экспорты сохранены. Время обхода сервиса и время получения различаются: кэш и юридическая актуальность требуют проверки. Демонстрационные вымышленные файлы находятся только в tests/fixtures/demo/ и не индексируются. Происхождение — DOCUMENT_PROVENANCE.md; исследование исходного проекта — reports/REPOSITORY_REVIEW.md.", "",
              "## Краткая архитектура RAG", "",
              "v1: вопрос → Gemini без документов. v2: вопрос → MCP-клиент → отдельный stdio MCP-сервер search_docs → сохранённый NumPy-индекс → top-3 выше порога → Gemini → проверка JSON, ссылок и чисел → ответ. Retrieval вызывается приложением до генерации, а не по необязательному решению LLM.", "",
              "И вопросы, и документы кодируются одной моделью. Для multilingual-e5-small используются рекомендованные префиксы query: и passage:. К тексту фрагмента при векторизации добавляется реальный заголовок родительского документа и контекст официального источника КФУ/ИТИС; в выдаче сохраняется исходный текст со смещениями. При превышении лимита токенов берётся взвешенное среднее окон; затем L2-нормализация. Это не обучение модели.", "",
              "## E1: метод и результаты", ""]
    if e1:
        report += [f"Полная матрица 6×6 — reports/e1_similarity.csv. Пары 1–2, 3–4, 5–6: {', '.join(f'{x:.4f}' for x in e1['pair_scores'])}.", "", e1["explanation"], "", "Ожидания сохранены перед вычислениями в reports/e1_expectations.json."]
    else:
        report += ["Не измерено: локальная модель ещё не запускалась."]
    report += ["", "## E2: количество и примеры фрагментов", "", f"Сохранено фрагментов: {len(chunks)}. Базовые параметры: 800 символов, перекрытие 150. Смещения start/end — по Unicode-символам, индекс фрагмента начинается с 0. Полные данные — index/chunks.json."]
    if e2:
        report += ["", "Три случайных примера выбраны с seed=42:"]
        for example in e2["examples"]:
            report += ["", f"### {example['id']} ({example['start']}–{example['end']})", "", "```text", example["text"], "```"]
        report += ["", "Нарезка с фиксированным шагом сохраняет точное перекрытие, но может обрывать слова и заголовки. Просмотр примеров не равен формальной гарантии осмысленности; для сдачи студенту следует их прочитать. Режим по абзацам не реализован: приоритет отдан точному базовому заданию."]
        review = optional("reports/e2_review.json")
        if review:
            report += ["", "Проверка содержательности трёх примеров (reports/e2_review.json):"]
            report += [f"- {r['id']}: {r['notes']}" for r in review['samples']]
    report += ["", "## E3: Recall@3", "", f"Фактический Recall@3: {e3['hits']}/15 = {e3['recall_at_3']:.3f}. Неудачные ID: {e3['failed_ids']}." if e3 else "Не измерено.", "",
               "По условию задания это доля вопросов, где хотя бы один эталонный фрагмент найден в top-3. Стандартный информационный recall с несколькими релевантными фрагментами определяется иначе. Эталоны получены по заранее записанным цитатам до поиска; вопросы E3 отдельны от E5."]
    baseline = optional("reports/baseline_minilm/recall.json")
    if baseline:
        report += ["", f"Исходная MiniLM дала {baseline['hits']}/15 на тех же эталонах. Прогон — reports/baseline_minilm/. E5-small без заголовков также дала 9/15 (reports/baseline_e5_800/); уменьшение размера до 500/350/200 само по себе не обеспечило цель (reports/retrieval_tuning.json). Сохранение реального заголовка родительского документа при векторизации дало 13/15 при 800/150 и 13/15 при 500/150; выбран больший размер по условию задания (reports/context_tuning.json). Тексты, вопросы и эталоны не менялись. Это подбор на малом E3-наборе; для независимой оценки нужны дополнительные вопросы."]
    if threshold:
        report += ["", f"Порог = {threshold['threshold']:.4f}; balanced accuracy калибровки = {threshold['balanced_accuracy']:.3f}. Положительных примеров пропущено {10-threshold['true_positive']}, отрицательных ложно принято {10-threshold['true_negative']}. Подробные score и кандидаты — reports/threshold.json.", "", threshold["limitation"]]
        report += ["", threshold['method'], "", "Первый максимум balanced accuracy без ограничения recall оставлял 8/10 положительных и 9/10 отрицательных (threshold_initial.json). Для снижения ложных отказов выбрано ограничение positive recall >= 0.9; это осознанный компромисс, а не доказательство, что каждый принятый фрагмент содержит ответ."]
    report += ["", "## E4: MCP и примеры с источниками", "",
               f"Локальная демонстрация MCP: {'выполнена, '+str(len(mcp_local['details']))+' вопросов; протокол initialize → list_tools → call_tool(search_docs), отдельный процесс' if mcp_local else 'ещё не выполнена'}. Данные — reports/mcp_local_demo.json.", "",
               "Локальная проверка MCP не является проверкой ответа Gemini. Реальные ответы E4 формируются командой demo-api: пять вопросов с ответом и два без ответа. До реального прогона и ручной проверки генеративная часть E4 не подтверждена. В unit-тестах используются явно отмеченные заглушки; их ответы не включаются в E4/E5.", "",
               "Валидация запрещает несуществующие номера, требует ссылки в каждом предложении v2 и сверяет цифры с указанными фрагментами. Отказ стандартизирован. Эти проверки НЕ доказывают смысловое подтверждение утверждений; фамилии, отношения и перефразирования необходимо проверять вручную.", "",
               "## E5: сравнительная таблица", "", f"Запуск: {name}. Получено реальных ответов: v1={metrics['v1']['completed']}, v2={metrics['v2']['completed']}. Полный эксперимент требует 50 в каждой версии. Значения ниже относятся только к завершённым вопросам; незаполненная ручная разметка обозначена «не измерено».", "",
               "| Метрика (цель — 50 вопросов) | v1 | v2 |", "|---|---|---|", *rows, "",
               "Вопросы: eval/questions_50.json, ровно 40 с ответом и 10 без него. Модель google/gemini-2.5-flash, температура 0 в обеих версиях. Результаты каждого вызова и usage сохраняются до валидации. При повторе запуска используются уже полученные ответы. Неудачные сетевые запросы могли быть обработаны провайдером: usage по ним может быть неизвестен.", "",
               "Разметка: eval/runs/<run>/manual_review.csv. Поля *_manual: 1 да, 0 нет, пусто не проверено. Для неприменимого критерия ставьте 0 и пояснение в notes (например, поддержка источником при отсутствии ссылок). Не включайте fallback в число правильных ответов Gemini: отмечайте origin и отдельно анализируйте качество итоговой системы.", "",
               "## Анализ ошибок", "", f"Вручную отмеченных ошибок v2: {len(errors)}. Детали — reports/error_analysis.json. Пока ручные метки не выставлены, отсутствие записей не означает отсутствие ошибок.", "",
               "Для неверного answerable-ответа наличие эталона в фактически переданном контексте указывает на вероятную ошибку генерации; отсутствие — на retrieval или порог. Для вопросов без ответа проверяется обоснованность отказа. Небольшой выбранный корпус официальных снимков не представляет все регламенты КФУ.", "",
               "## Цена использования LLM", "", "Стоимость = (input_tokens × тариф_входа + output_tokens × тариф_выхода) / 1 000 000. Прогноз на N вопросов = средняя стоимость наблюдённого вопроса × N. Повторные попытки формата включены в usage; неизвестные значения не заменяются нулём."]
    for version in ["v1", "v2"]:
        m = metrics[version]
        cost = cost_projection(m["input_total"], m["output_total"], m["completed"], target_questions, price_in, price_out)
        report += ["", f"{version}: фактические входные токены {measured(m['input_total'])}, выходные {measured(m['output_total'])}; прогноз на {target_questions} вопросов: {measured(cost, ' USD')}."]
        report += ["", f"Правильных ответов: {measured(100*m['correct']/m['completed'], '%') if m['completed'] and m['correct'] is not None else 'не измерено'}; корректных отказов: {measured(m['correct_refusal'])}. Ручных меток правильности: {m['correct_reviewed']}/{m['completed']}."]
    a, b = metrics['v1'], metrics['v2']
    halluc_delta = b['hallucinated'] - a['hallucinated'] if a['completed'] == b['completed'] and a['hallucinated'] is not None and b['hallucinated'] is not None else None
    token_ratio = b['input_per_question'] / a['input_per_question'] if a['completed'] == b['completed'] and a['input_per_question'] and b['input_per_question'] is not None else None
    report += ["", f"Изменение числа галлюцинаций v2−v1: {measured(halluc_delta)}. Отношение входных токенов v2/v1: {measured(token_ratio)}.", "", "## Отдельный инструмент расписания", "", "get_group_schedule проксирует рабочий JS MCP-Demo через официальный Python MCP-клиент. Прямая свежая Google-таблица доступна; 11-307/понедельник сверено с опубликованным официальным PDF четвёртого курса 2026/27. Другие ячейки не сверены автоматически. В выдаче видны fetchedAt, mode, source и verified_against_official_pdf. Недели, подгруппы, праздники и отмены требуют проверки. Источник и измерения отделены от search_docs; расписание не участвует в E5. Журналы — reports/reference_schedule_check.json, reports/schedule_mcp_check.json; область проверки — data/schedule/verification.json."]
    report += ["", f"Исходный бюджет ДЗ 2: {os.getenv('HW2_BUDGET') or 'не предоставлен (незаполненное поле)'}. Тарифы не придуманы: INPUT_PRICE_PER_MILLION и OUTPUT_PRICE_PER_MILLION заполняются по актуальному провайдеру.", "",
               "## Проверки и ограничения", "", (f"pytest: {tests['passed']} passed из {tests['tests']}, failures={tests['failures']}, errors={tests['errors']}, skipped={tests['skipped']}; настоящая модель: {tests['real_local_model']}." if tests else "pytest пока не выполнен; см. PROGRESS.md") + " Полный журнал — reports/pytest_output.txt. В исходном JS MCP-Demo отдельно прошли 15 тестов, его LLM в unit-тестах подменяется.", "",
               "Без ключа не подтверждены реальные ответы Gemini, его фактическая модель/usage, отказ на тематически близкие неизвестные вопросы, стоимость и сравнение качества 50×2. Порог не гарантирует наличие ответа. Семантическая проверка источников требует человека. Не реализован дополнительный режим нарезки по абзацам.", "",
               "## Вывод", "", "Проект разделяет воспроизводимые локальные измерения и проверки, требующие платного API и ручной разметки. Утверждать уменьшение галлюцинаций можно только после завершённого и проверенного сравнения, а не по архитектуре RAG самой по себе."]
    (reports / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (reports / "ERROR_ANALYSIS.md").write_text("# Анализ ошибок v2\n\n" + ("\n\n".join(f"- {r['id']}: {r['reason']}" for r in errors) or "Ошибки пока не размечены. Это не доказательство безошибочности.") + "\n", encoding="utf-8")
    create_review_template()
    generate_e4_demo()
    return metrics


def generate_e4_demo():
    records = load_results(run_path("e4_demo"))
    labels = {(r['id'], r['version']): r for r in export_review(run_path('e4_demo'))} if records else {}
    text = ["# E4: демонстрация ответов и источников", "", "Команда: python -m src.cli demo-api. Пять вопросов с ответом и два без ответа. Корректность подтверждается студентом по эталону.", ""]
    if not records:
        text += ["Реальный Gemini не запускался: API-ключ не предоставлен. Ответы, источники ответа и корректность не измерены. Локальные MCP-вызовы находятся в reports/mcp_local_demo.json; они не подменяют генерацию."]
    for r in records:
        text += [f"## {r['id']}: {r['question']}", "", "Вызов: search_docs(query=вопрос), настоящий stdio MCP.", ""]
        for h in (r['retrieval'] or {}).get('fragments', []):
            text += [f"Источник [{h['number']}]: {h['id']}, score={h['score']:.4f}, URL: {h.get('url', 'не указан')}", "", "```text", h['text'], "```", ""]
        text += ["Ответ: " + r['answer']['answer'], "", f"sources={r['answer']['sources']}, found={r['answer']['found']}, origin={r['origin']}", "", "Эталон: " + r['reference']['reference_answer'], "", "Корректность (ручная метка 0/1): " + (labels.get((r['id'], r['version']), {}).get('correct_manual') or 'не измерено'), ""]
    (ROOT / 'reports/E4_DEMO.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
