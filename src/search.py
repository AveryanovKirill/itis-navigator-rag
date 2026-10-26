import os
from .config import ROOT
from .embeddings import LocalEmbeddings
from .indexing import VectorIndex
from .storage import read_json, fingerprint


def load_search(directory=None, docs_dir=None):
    directory = directory or ROOT / "index"
    from pathlib import Path
    missing = [name for name in ("chunks.json", "metadata.json", "embeddings.npy") if not (Path(directory) / name).exists()]
    if missing:
        raise ValueError("Индекс отсутствует или неполон: " + ", ".join(missing) + ". Добавьте UTF-8 документы в docs/ и выполните python -m src.bootstrap.")
    return VectorIndex(directory, LocalEmbeddings(), docs_dir=docs_dir or ROOT / "docs")


def relevance_threshold(index):
    metadata = index if isinstance(index, dict) else index.metadata
    override = os.getenv("THRESHOLD", "").strip()
    if override:
        value = float(override)
    else:
        path = ROOT / "reports/threshold.json"
        if not path.exists():
            raise ValueError("Порог ещё не откалиброван. Выполните python -m src.cli calibrate.")
        calibration = read_json(path)
        if calibration["index_hash"] != fingerprint(metadata):
            raise ValueError("Калибровка устарела; выполните calibrate заново.")
        value = calibration["threshold"]
    if not -1 <= value <= 1:
        raise ValueError("THRESHOLD должен быть между -1 и 1.")
    return value


def search_payload(index, query, threshold):
    raw = index.search(query, k=3, threshold=-1)
    hits = [h for h in raw if h['score'] >= threshold]
    return {"status": "found" if hits else "not_found", "query": query,
            "threshold": threshold, "fragments": hits,
            "candidates_before_threshold": [{"id": h['id'], "score": h['score']} for h in raw],
            "message": "Найдены фрагменты; проверьте, содержат ли они ответ." if hits else "Подходящие документы не найдены."}
