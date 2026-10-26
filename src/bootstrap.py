"""Подготовка локального корпуса и индекса; готовый индекс не перестраивается."""
import argparse
import asyncio
import os
import sys
from pathlib import Path
import httpx

from .config import ROOT, OLLAMA_BASE_URL, OLLAMA_MODEL


async def check_ollama():
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            version = await client.get(OLLAMA_BASE_URL + "/api/version")
            version.raise_for_status()
            tags = await client.get(OLLAMA_BASE_URL + "/api/tags")
            tags.raise_for_status()
    except httpx.HTTPError as exc:
        raise ValueError(f"Ollama недоступна по адресу {OLLAMA_BASE_URL}. Запустите установленное приложение Ollama или ollama serve. Проверьте OLLAMA_BASE_URL в .env.") from exc
    names = {m["name"] for m in tags.json().get("models", [])}
    if OLLAMA_MODEL not in names:
        raise ValueError("Модель qwen3:4b не найдена. Выполните ollama pull qwen3:4b. Другие модели автоматически не выбираются.")
    return {"status": "ok", "base_url": OLLAMA_BASE_URL, "ollama_version": version.json().get("version"),
            "model": OLLAMA_MODEL, "generation_performed": False}


def prepare(fetch_official=False, download_only=False, rebuild=False):
    from .embeddings import LocalEmbeddings
    from .chunking import document_paths, make_chunks
    from .indexing import build_index, VectorIndex
    from .search import relevance_threshold
    for name in ("docs", "index", "data", "reports", "eval"):
        (ROOT / name).mkdir(exist_ok=True)
    if fetch_official:
        if document_paths(ROOT / "docs"):
            raise ValueError("docs/ уже содержит документы. Автоматическая загрузка не перезаписывает их; выберите новую копию проекта для обновления корпуса.")
        from .official import import_official
        import_official(reuse_downloads=True)
    if not download_only and not document_paths(ROOT / "docs"):
        raise ValueError("В docs/ нет документов. Сохраните разрешённые UTF-8 .txt/.md и метаданные data/sources.json либо выполните python -m src.bootstrap --fetch-official. Полные копии материалов КФУ не распространяются с кодом.")
    embedder = LocalEmbeddings()
    if download_only:
        print("Эмбеддинги доступны локально: " + str(embedder.signature))
        return
    complete = all((ROOT / "index" / f).exists() for f in ("chunks.json", "metadata.json", "embeddings.npy"))
    if complete and not rebuild:
        index = VectorIndex(ROOT / "index", embedder, docs_dir=ROOT / "docs")
        relevance_threshold(index)
        print("Готовый индекс проверен и сохранён без перестроения.")
        return
    # В публикуемой копии нет старой калибровки: стартовый порог задаётся явно в .env.
    if not os.getenv("THRESHOLD", "").strip() and not (ROOT / "reports/threshold.json").exists():
        raise ValueError("Задайте THRESHOLD в .env (стартовый пример: 0.817020058631897). Для другого корпуса проверьте порог на своих положительных и отрицательных запросах.")
    chunks, _ = make_chunks(ROOT / "docs", ROOT / "index/chunks.json", 800, 150)
    build_index(ROOT / "index/chunks.json", ROOT / "index", embedder, ROOT / "docs")
    print(f"Индекс подготовлен: {len(chunks)} фрагментов, 800/150; векторы сохранены в index/.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-official", action="store_true", help="Получить официальные страницы для локального использования; только при пустой docs/")
    parser.add_argument("--download-embeddings-only", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true", help="Явно разрешить перестроение индекса этой копии проекта")
    args = parser.parse_args()
    try:
        prepare(args.fetch_official, args.download_embeddings_only, args.rebuild_index)
    except (ValueError, RuntimeError, OSError) as exc:
        print("Подготовка не завершена: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
