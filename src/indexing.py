from pathlib import Path
import hashlib
import numpy as np

from .chunking import read_documents
from .embeddings import normalize
from .storage import fingerprint, read_json, save_json


def build_index(chunks_path, directory, embedder, docs_dir, contextual_title=True):
    chunks = read_json(chunks_path)
    if not chunks:
        raise ValueError("Нет фрагментов для индексации.")
    docs = read_documents(docs_dir)
    for c in chunks:
        if docs.get(c["source"], "")[c["start"]:c["end"]] != c["text"]:
            raise ValueError("Документы изменились. Сначала повторите chunk.")
    # Короткие контактные/списочные фрагменты теряют название родительского документа.
    # Добавляем только его реальный заголовок, без вопроса, эталона или LLM-генерации.
    texts = []
    for c in chunks:
        title = docs[c["source"]].splitlines()[0].lstrip("# ").strip()
        institution = "КФУ, ИТИС. " if c.get("url", "").startswith("https://kpfu.ru/") else ""
        texts.append(f"{institution}{title}\n{c['text']}" if contextual_title else c['text'])
    vectors = normalize(embedder.encode(texts))
    if len(vectors) != len(chunks) or np.any(np.linalg.norm(vectors, axis=1) < .99):
        raise ValueError("Модель вернула пустые или некорректные эмбеддинги.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "embeddings.npy.tmp").open("wb") as f:
        np.save(f, vectors, allow_pickle=False)
    (directory / "embeddings.npy.tmp").replace(directory / "embeddings.npy")
    metadata = {"embedding": embedder.signature, "chunks_hash": fingerprint(chunks),
                "documents_hash": fingerprint(docs), "shape": list(vectors.shape),
                "text_template": "official_institution_document_title_then_chunk_v1" if contextual_title else "plain_chunk_v1",
                "vectors_sha256": hashlib.sha256((directory / "embeddings.npy").read_bytes()).hexdigest()}
    save_json(directory / "metadata.json", metadata)
    return metadata


class VectorIndex:
    def __init__(self, directory, embedder, docs_dir=None):
        directory = Path(directory)
        self.chunks = read_json(directory / "chunks.json")
        self.metadata = read_json(directory / "metadata.json")
        if self.metadata["embedding"] != embedder.signature:
            raise ValueError("Модель/алгоритм эмбеддингов отличаются от индекса. Перестройте индекс.")
        if fingerprint(self.chunks) != self.metadata["chunks_hash"]:
            raise ValueError("chunks.json изменён; перестройте индекс.")
        if docs_dir and fingerprint(read_documents(docs_dir)) != self.metadata["documents_hash"]:
            raise ValueError("Документы изменены; выполните chunk, build, recall и calibrate заново.")
        if hashlib.sha256((directory / "embeddings.npy").read_bytes()).hexdigest() != self.metadata.get("vectors_sha256"):
            raise ValueError("Матрица индекса изменена/устарела; перестройте индекс.")
        self.vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
        if list(self.vectors.shape) != self.metadata["shape"] or len(self.vectors) != len(self.chunks) or not np.isfinite(self.vectors).all():
            raise ValueError("Повреждена матрица индекса.")
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-4):
            raise ValueError("Векторы индекса должны быть нормализованы.")
        self.embedder = embedder

    def search(self, query, k=3, threshold=-1.0):
        if not 1 <= k <= 3 or not -1 <= threshold <= 1:
            raise ValueError("k должен быть 1–3; threshold — от -1 до 1.")
        if not query.strip() or not len(self.chunks):
            return []
        encode_query = getattr(self.embedder, "encode_query", self.embedder.encode)
        vector = normalize(encode_query([query]))[0]
        scores = self.vectors @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        hits = [dict(self.chunks[int(i)], score=float(scores[i])) for i in order if scores[i] >= threshold]
        return [dict(hit, number=i + 1) for i, hit in enumerate(hits)]
