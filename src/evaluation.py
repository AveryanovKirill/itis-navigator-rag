import csv
from contextlib import asynccontextmanager
from pathlib import Path

from .agent import create_llm, ask, APIError
from .config import ROOT, MODEL
from .exercises import validate_dataset, gold_ids
from .mcp_client import connect_mcp, call_search
from .storage import read_json, save_json, fingerprint, now


@asynccontextmanager
async def no_mcp():
    yield None


def run_path(name):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Имя --run: латинские буквы, цифры, _ и -.")
    return ROOT / "eval/runs" / name


async def experiment(name="main", limit=50, demo=False):
    llm = create_llm()
    directory = run_path(name)
    questions = read_json(ROOT / "eval/questions_50.json")
    chunks = read_json(ROOT / "index/chunks.json")
    gold = validate_dataset(questions, chunks)
    if not 1 <= limit <= 50:
        raise ValueError("limit должен быть 1–50.")
    if demo:
        selected = [questions[i] for i in [8, 16, 24, 12, 28, 40, 41]]
    else:
        selected = questions[:limit]
    from .search import relevance_threshold
    from .embeddings import EMBEDDING_MODEL
    # Индекс и документы проверяет сервер; manifest фиксирует его содержимое.
    metadata = read_json(ROOT / "index/metadata.json")
    threshold = relevance_threshold(metadata)
    settings = {"model": llm.model, "provider": llm.backend, "temperature": 0, "dataset_hash": fingerprint(questions), "index_hash": fingerprint(metadata),
                "threshold": threshold, "embedding_model": EMBEDDING_MODEL, "mode": "demo_e4" if demo else "e5",
                "prompt_hashes": {v: fingerprint((ROOT / f"prompts/system_{v}.md").read_text(encoding="utf-8")) for v in ["v1", "v2"]},
                "protocol": "forced MCP retrieval before generation", "format_attempts": 2, "api_attempts": 3,
                "generation_options": getattr(llm, "options", {"max_tokens": 1800}), "think": False if llm.backend == "ollama" else None}
    settings["implementation_hash"] = fingerprint({name: (ROOT / 'src' / name).read_text(encoding='utf-8') for name in ['agent.py', 'schemas.py', 'citations.py', 'evaluation.py', 'mcp_server.py', 'search.py', 'indexing.py', 'embeddings.py']})
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["settings"] != settings:
        raise ValueError("Корпус, вопросы, prompts или настройки изменились. Задайте новое имя --run.")
    if not manifest_path.exists():
        save_json(manifest_path, {"created_at": now(), "settings": settings})
    save_json(directory / "gold_chunks.json", gold)
    versions = ["v2"] if demo else ["v1", "v2"]
    # Один сервер на весь пакет: эмбеддинги документов не пересчитываются.
    async with connect_mcp() as session:
        for q in selected:
            for version in versions:
                target = directory / f"{q['id']}_{version}.json"
                if target.exists():
                    print(f"{q['id']} {version}: уже сохранён, пропуск.")
                    continue
                checkpoint_dir = directory / "calls" / f"{q['id']}_{version}"
                retrieval = None
                if version == "v2":
                    retrieval_path = checkpoint_dir / "retrieval.json"
                    retrieval = read_json(retrieval_path) if retrieval_path.exists() else await call_search(session, q["question"])
                    save_json(retrieval_path, retrieval)
                try:
                    result = await ask(q["question"], version, llm, mcp=session if version == "v2" else None,
                                       checkpoint_dir=checkpoint_dir, retrieval=retrieval)
                except APIError as exc:
                    save_json(directory / "last_error.json", {"id": q["id"], "version": version, "error": str(exc), "timestamp": now()})
                    raise
                result["id"] = q["id"]
                result["reference"] = q  # Только после вызова LLM.
                result["gold_chunks"] = gold[q["id"]]
                save_json(target, result)
                print(f"{q['id']} {version}: {result['origin']}; найдено={result['answer']['found']}; input={result['usage']['prompt_tokens']}; секунд={result['elapsed_seconds']:.2f}", flush=True)
    export_review(directory)
    return directory


REVIEW_FIELDS = ["id", "version", "question", "answerable", "reference_answer", "evidence", "reference_source", "answer",
                 "found", "origin", "sources", "retrieved_fragments", "retrieval_hit", "retrieval_hit_before_threshold", "has_source_auto", "input_tokens",
                 "correct_auto_hint", "correct_manual", "hallucinated_manual", "source_support_manual", "refusal_correct_manual", "notes"]


def load_results(directory):
    return [read_json(p) for p in sorted(Path(directory).glob("q*_v[12].json"))]


def export_review(directory):
    directory = Path(directory)
    path = directory / "manual_review.csv"
    previous = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            previous = {(r["id"], r["version"]): r for r in csv.DictReader(f)}
    rows = []
    for result in load_results(directory):
        q = result["reference"]
        answer = result["answer"]
        hits = (result["retrieval"] or {}).get("fragments", [])
        hit = bool(set(result["gold_chunks"]) & {h["id"] for h in hits})
        raw_hit = bool(set(result["gold_chunks"]) & {h["id"] for h in (result["retrieval"] or {}).get("candidates_before_threshold", [])})
        row = {"id": result["id"], "version": result["version"], "question": q["question"], "answerable": q["answerable"],
               "reference_answer": q["reference_answer"], "evidence": q["evidence"], "reference_source": q["source"],
               "answer": answer["answer"], "found": answer["found"], "origin": result["origin"],
               "sources": str(answer["sources"]), "retrieved_fragments": "\n\n".join(f"[{h['number']}] {h['id']} score={h['score']:.4f}\n{h['text']}" for h in hits),
               "retrieval_hit": hit if q["answerable"] and result["version"] == "v2" else "",
               "retrieval_hit_before_threshold": raw_hit if q["answerable"] and result["version"] == "v2" else "",
               "has_source_auto": bool(answer["sources"]), "input_tokens": result["usage"]["prompt_tokens"],
               "correct_auto_hint": "Возможен верный отказ; проверьте вручную" if not q["answerable"] and not answer["found"] else "Требует ручной проверки"}
        old = previous.get((row["id"], row["version"]), {})
        for field in ["correct_manual", "hallucinated_manual", "source_support_manual", "refusal_correct_manual", "notes"]:
            row[field] = old.get(field, "")
        rows.append(row)
    directory.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    return rows


def parse_label(value):
    if value == "":
        return None
    if str(value) not in {"0", "1"}:
        raise ValueError("Ручные метки должны быть 0, 1 или пустыми.")
    return int(value)


def aggregate(results, reviews):
    metrics = {}
    labels = {(r["id"], r["version"]): r for r in reviews}
    for version in ["v1", "v2"]:
        subset = [r for r in results if r["version"] == version]
        rows = [labels.get((r["id"], version), {}) for r in subset]
        m = {"completed": len(subset), "has_source": sum(bool(r["answer"]["sources"]) for r in subset) if subset else None}
        for metric, field in [("correct", "correct_manual"), ("hallucinated", "hallucinated_manual"),
                              ("source_supported", "source_support_manual"), ("correct_refusal", "refusal_correct_manual")]:
            values = [parse_label(r.get(field, "")) for r in rows]
            m[metric] = sum(v for v in values if v is not None) if values and all(v is not None for v in values) else None
            m[f"{metric}_reviewed"] = sum(v is not None for v in values)
        tokens = [r["usage"]["prompt_tokens"] for r in subset]
        output = [r["usage"]["completion_tokens"] for r in subset]
        m["input_total"] = sum(tokens) if tokens and all(t is not None for t in tokens) else None
        m["output_total"] = sum(output) if output and all(t is not None for t in output) else None
        m["input_per_question"] = m["input_total"] / len(subset) if m["input_total"] is not None else None
        metrics[version] = m
    return metrics


def error_analysis(results, reviews):
    labels = {(r["id"], r["version"]): r for r in reviews}
    rows = []
    for r in results:
        if r["version"] != "v2" or labels.get((r["id"], "v2"), {}).get("correct_manual") != "0":
            continue
        hits = (r["retrieval"] or {}).get("fragments", [])
        hit = bool(set(r["gold_chunks"]) & {h["id"] for h in hits})
        raw_hit = bool(set(r["gold_chunks"]) & {h["id"] for h in (r["retrieval"] or {}).get("candidates_before_threshold", [])})
        if not r["reference"]["answerable"]:
            reason = "Неподтверждённый ответ/неверный отказ на вопрос без ответа; проверьте использование контекста."
        elif hit:
            reason = "Вероятная ошибка генерации, промпта или использования найденного контекста."
        elif raw_hit:
            reason = "Нужный фрагмент был в сыром top-3, но отсеян порогом: ошибка фильтрации релевантности."
        else:
            reason = "Нужного фрагмента нет в сыром top-3: вероятная ошибка retrieval."
        rows.append({"id": r["id"], "question": r["question"], "retrieval_hit": hit, "retrieval_hit_before_threshold": raw_hit, "reason": reason})
    return rows
