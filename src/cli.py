import argparse
import asyncio
import json
import sys
import httpx
from .config import ROOT, OLLAMA_BASE_URL
from .storage import save_json, read_json, now


def display(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


async def mcp_demo(question=None):
    from .mcp_client import connect_mcp, call_search
    if question:
        selected = [question]
    elif (ROOT / "eval/questions_50.json").exists():
        questions = read_json(ROOT / "eval/questions_50.json")
        selected = [questions[i]["question"] for i in [0, 8, 16, 24, 32, 40, 41]] + ["Как испечь яблочный пирог?"]
    else:
        selected = ["К какому заместителю директора ИТИС обращаться по академическим стипендиям?", "Как испечь яблочный пирог?"]
    details = []
    async with connect_mcp() as session:
        for q in selected:
            data = await call_search(session, q)
            details.append(data)
            display(data)
    save_json(ROOT / ("reports/mcp_check.json" if question else "reports/mcp_local_demo.json"),
              {"transport": "stdio", "real_subprocess": True, "timestamp": now(), "details": details, "gemini_called": False})


async def ask_cli(version, question, group=None):
    from .schedule import schedule_intent, schedule_args
    if version == "v2" and schedule_intent(question):
        group, day = schedule_args(question, group)
        # Расписание выводится без пересказа LLM; источник и fetchedAt видны пользователю.
        from .mcp_client import connect_mcp
        async with connect_mcp() as session:
            result = await session.call_tool("get_group_schedule", {"group": group, "day": day})
            data = result.structuredContent or json.loads("\n".join(c.text for c in result.content if c.type == "text"))
            save_json(ROOT / "reports/schedule_route_check.json", {"question": question, "arguments": {"group": group, "day": day}, "gemini_called": False, "data": data, "timestamp": now()})
            display({"route": "schedule", "gemini_called": False, "result": data})
            print(data.get("formattedAnswer", data.get("message", "Нет записей в таблице.")))
        return
    from .agent import create_llm, ask
    llm = create_llm()
    if version == "v2":
        from .mcp_client import connect_mcp
        async with connect_mcp() as session:
            result = await ask(question, version, llm, mcp=session)
    else:
        result = await ask(question, version, llm)
    display(result)
    for hit in (result["retrieval"] or {}).get("fragments", []):
        if hit["number"] in result["answer"]["sources"]:
            print(f"[{hit['number']}] {(ROOT / 'docs' / hit['source']).resolve()} (фрагмент {hit['chunk_index']}, score={hit['score']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="ИТИС Навигатор — RAG по официальным снимкам КФУ, E1–E5.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("e1", help="Эмбеддинги 6 фраз и матрица 6×6")
    chunk = sub.add_parser("chunk", help="E2: нарезка документов")
    chunk.add_argument("--chunk-size", type=int, default=800)
    chunk.add_argument("--overlap", type=int, default=150)
    sub.add_parser("build", help="Построение сохранённого NumPy-индекса")
    search = sub.add_parser("search", help="Поиск top-3 по сохранённому индексу")
    search.add_argument("query")
    search.add_argument("--raw", action="store_true", help="Без порога, как в E3")
    sub.add_parser("recall", help="E3: Recall@3 на 15 вопросах")
    sub.add_parser("calibrate", help="Подбор порога по отдельным вопросам")
    mcp = sub.add_parser("mcp-check", help="Настоящий MCP initialize/list_tools/call_tool без LLM")
    mcp.add_argument("query", nargs="?")
    ask = sub.add_parser("ask", help="Реальная LLM: локальная Ollama по умолчанию")
    ask.add_argument("version", choices=["v1", "v2"])
    ask.add_argument("question")
    ask.add_argument("--group", help="Группа для вопроса о расписании")
    schedule = sub.add_parser("schedule", help="Отдельный MCP-инструмент расписания; без Gemini")
    schedule.add_argument("group", nargs="?", default="")
    schedule.add_argument("--day", default="")
    sub.add_parser("import-official", help="Обновить пять официальных снимков через прямой HTTPS; эталоны после этого проверить заново")
    exp = sub.add_parser("experiment", help="E5: реальный API-прогон с продолжением")
    exp.add_argument("--limit", type=int, default=50)
    exp.add_argument("--run", default="ollama_rag_citations")
    demo = sub.add_parser("demo-api", help="E4: 5 известных и 2 неизвестных вопроса к LLM v2")
    demo.add_argument("--run", default="ollama_e4_citations")
    report = sub.add_parser("report", help="Отчёт по реальным данным; неизвестные метрики не измерены")
    report.add_argument("--run", default="ollama_rag_citations")
    report.add_argument("--target-questions", type=int, default=1000)
    previous = sub.add_parser("previous-experiment", help="Предыдущая практика: v0/v1 с одинаковыми инструментами")
    previous.add_argument("--run", default="ollama_prompts")
    previous.add_argument("--limit", type=int, default=50)
    previous_report = sub.add_parser("previous-report", help="Отчёт предыдущей практики по ручным CSV-меткам")
    previous_report.add_argument("--run", default="ollama_prompts")
    previous_ask = sub.add_parser("previous-ask", help="Реальный ответ прошлой практики, например p06")
    previous_ask.add_argument("version", choices=["v0", "v1"])
    previous_ask.add_argument("id")
    sub.add_parser("ollama-check", help="Проверить локальную Qwen3:4b одним реальным запросом и показать GPU")
    sub.add_parser("doctor", help="Проверить Ollama и наличие Qwen без генерации")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            from .bootstrap import check_ollama
            display(asyncio.run(check_ollama()))
        elif args.command == "chunk":
            from .chunking import make_chunks
            chunks, examples = make_chunks(ROOT / "docs", ROOT / "index/chunks.json", args.chunk_size, args.overlap)
            save_json(ROOT / "reports/e2_results.json", {"count": len(chunks), "chunk_size": args.chunk_size, "overlap": args.overlap, "seed": 42, "examples": examples})
            print(f"Создано {len(chunks)} фрагментов. Случайные примеры (seed=42):")
            for c in examples:
                print(f"\n{c['id']} | {c['start']}–{c['end']}\n{c['text']}")
        elif args.command in {"e1", "build"}:
            from .embeddings import LocalEmbeddings
            model = LocalEmbeddings()
            if args.command == "e1":
                from .exercises import e1
                e1(model)
            else:
                from .indexing import build_index
                display(build_index(ROOT / "index/chunks.json", ROOT / "index", model, ROOT / "docs"))
        elif args.command in {"search", "recall", "calibrate"}:
            from .search import load_search, relevance_threshold, search_payload
            from .exercises import recall, calibrate
            index = load_search()
            if args.command == "search":
                display(search_payload(index, args.query, -1 if args.raw else relevance_threshold(index)))
            elif args.command == "recall":
                recall(index)
            else:
                calibrate(index)
        elif args.command == "mcp-check":
            asyncio.run(mcp_demo(args.query))
        elif args.command == "ask":
            asyncio.run(ask_cli(args.version, args.question, args.group))
        elif args.command == "schedule":
            from .schedule import check_schedule
            data = asyncio.run(check_schedule(args.group, args.day))
            display(data)
            print(data.get("formattedAnswer", data.get("message", "Нет записей в таблице.")))
        elif args.command == "import-official":
            from .official import import_official
            import_official()
        elif args.command in {"experiment", "demo-api"}:
            from .evaluation import experiment
            directory = asyncio.run(experiment(args.run, getattr(args, "limit", 50), demo=args.command == "demo-api"))
            print(f"Результаты: {directory}")
            print("Заполните manual_review.csv и затем выполните report --run " + args.run)
        elif args.command == "report":
            from .config import LLM_PROVIDER
            if LLM_PROVIDER == "ollama":
                from .completion_reports import final_report
                display(final_report(args.run))
            else:
                from .reporting import generate_report
                display(generate_report(args.run, args.target_questions))
            print("Отчёт: reports/FINAL_REPORT.md")
        elif args.command == "previous-experiment":
            from .previous_practice import experiment as previous_experiment
            print(asyncio.run(previous_experiment(args.run, args.limit)))
        elif args.command == "previous-report":
            from .completion_reports import previous_report
            display(previous_report(args.run))
        elif args.command == "previous-ask":
            from .previous_practice import BASE, tool_context, generate, PreviousAnswer
            from .agent import Ollama
            from .mcp_client import connect_mcp
            q = next((q for q in read_json(BASE / "questions_50.json") if q["id"] == args.id), None)
            if q is None:
                raise ValueError("Укажите ID из questions_50.json: p01–p50.")
            async def previous_live():
                import uuid
                async with connect_mcp() as session:
                    context = await tool_context(q, session)
                    result = await generate(q, args.version, context, Ollama(schema=PreviousAnswer.model_json_schema()),
                                            ROOT / "reports/demo_calls" / uuid.uuid4().hex)
                    save_json(ROOT / f"reports/previous_demo_{q['id']}_{args.version}.json", result)
                    display({"question": q["question"], "answer": result["answer"], "origin": result["origin"], "usage": result["usage"], "elapsed_seconds": result["elapsed_seconds"], "source": context})
            asyncio.run(previous_live())
        elif args.command == "ollama-check":
            from .agent import Ollama
            import httpx
            async def check_local():
                async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
                    version = (await client.get(OLLAMA_BASE_URL + "/api/version")).json()
                    tags = (await client.get(OLLAMA_BASE_URL + "/api/tags")).json()
                if "qwen3:4b" not in {m["name"] for m in tags.get("models", [])}:
                    raise ValueError("Установленная qwen3:4b не найдена. Автоматической загрузки нет.")
                reply = await Ollama().complete([{"role": "system", "content": 'Верни JSON: answer="В доступных документах недостаточно информации для ответа.", sources=[], found=false.'}, {"role": "user", "content": "Назови мою личную оценку за экзамен."}])
                async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
                    ps = (await client.get(OLLAMA_BASE_URL + "/api/ps")).json()
                data = {"timestamp": now(), "version": version, "tags": tags, "reply": reply, "ps": ps}
                save_json(ROOT / "reports/ollama_check.json", data)
                display(data)
            asyncio.run(check_local())
    except httpx.HTTPError as exc:
        print(f"Ошибка соединения с API: {exc}. Для локального режима запустите Ollama и проверьте http://127.0.0.1:11434.", file=sys.stderr)
        return 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except ExceptionGroup as exc:
        def leaves(group):
            return [leaf for e in group.exceptions for leaf in (leaves(e) if isinstance(e, BaseExceptionGroup) else [e])]
        message = "; ".join(str(e) for e in leaves(exc))
        print(f"Ошибка MCP: {message}. Проверьте индекс, калибровку и запуск сервера. Для диагностики: ITIS_TRACE_MCP=1.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
