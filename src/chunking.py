from pathlib import Path
import random

from .storage import save_json, read_json


def split_text(text, source, chunk_size=800, overlap=150):
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Нужны chunk_size > 0 и 0 <= overlap < chunk_size.")
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        i = len(chunks)
        chunks.append({"id": f"{source}:{i}", "source": source, "chunk_index": i,
                       "start": start, "end": end, "text": text[start:end]})
        if end == len(text):
            break
        start = end - overlap
    return chunks


def document_paths(directory):
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in {".txt", ".md"} and p.is_file())


def read_documents(directory):
    result = {}
    for path in document_paths(directory):
        if path.is_symlink():
            raise ValueError(f"Символические ссылки не поддерживаются: {path.name}")
        try:
            result[path.relative_to(directory).as_posix()] = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Не удалось прочитать UTF-8 документ {path.name}: {exc}") from exc
    if not result:
        raise ValueError("В docs/ нет .txt или .md файлов. Добавьте документы.")
    return result


def make_chunks(directory, output, chunk_size=800, overlap=150):
    docs = read_documents(directory)
    chunks = [c for source, text in docs.items() for c in split_text(text, source, chunk_size, overlap)]
    provenance_path = Path(directory).parent / "data/sources.json"
    provenance = read_json(provenance_path) if provenance_path.exists() else {}
    for c in chunks:
        if c["source"] in provenance:
            c.update({k: provenance[c["source"]][k] for k in ["url", "fetched_at", "validity"]})
    save_json(output, chunks)
    examples = random.Random(42).sample(chunks, min(3, len(chunks)))
    return chunks, examples
