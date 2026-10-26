import sys
import os
import numpy as np

from .config import EMBEDDING_MODEL, ROOT


def normalize(vectors):
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim not in (1, 2) or not np.isfinite(array).all():
        raise ValueError("Ожидаются конечные векторы размерности 1 или 2.")
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)


def cosine_matrix(a, b):
    return normalize(a) @ normalize(b).T


class LocalEmbeddings:
    def __init__(self, model_name=EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(4)
        self.model_name = model_name
        # Сторонние сообщения не должны загрязнять stdout MCP.
        print(f"Загрузка локальной модели {model_name}…", file=sys.stderr)
        cache = ROOT / ".cache/huggingface/hub"
        repo = cache / ("models--" + model_name.replace("/", "--"))
        ref = repo / "refs/main"
        revision = os.getenv("EMBEDDING_REVISION", "").strip() or (ref.read_text().strip() if ref.exists() else None)
        snapshot = repo / "snapshots" / revision if revision else None
        cached = snapshot is not None and (snapshot / "model.safetensors").exists() and (snapshot / "1_Pooling/config.json").exists()
        try:
            self.model = SentenceTransformer(str(snapshot) if cached else model_name, cache_folder=str(cache),
                device="cpu", trust_remote_code=False, local_files_only=cached, **({"revision": revision} if not cached and revision else {}))
        except (OSError, RuntimeError) as exc:
            raise RuntimeError("Не удалось загрузить эмбеддинги. Для первой загрузки нужен интернет; уберите HF_HUB_OFFLINE=1 и повторите python -m src.bootstrap --download-embeddings-only. Кэш хранится в .cache/huggingface внутри проекта.") from exc
        self.model.eval()
        dim = self.model.get_embedding_dimension() if hasattr(self.model, "get_embedding_dimension") else self.model.get_sentence_embedding_dimension()
        self.signature = {"model": model_name, "dimension": dim,
                          "max_seq_length": self.model.max_seq_length, "pooling": "token_windows_weighted_mean_v1"}
        self.signature["revision"] = revision or getattr(self.model[0].auto_model.config, "_commit_hash", None)
        self.signature["prefixes"] = "query:/passage:" if "multilingual-e5" in model_name else "none"

    def encode(self, texts, role="passage"):
        """Не теряем конец 800-символьного куска при лимите модели 128 токенов.

        Длинный текст: окна токенизатора без перекрытия → среднее эмбеддингов,
        взвешенное по числу содержательных токенов → нормализация.
        Один и тот же алгоритм применяется к документам и вопросам.
        """
        tokenizer = self.model.tokenizer
        width = self.model.max_seq_length - tokenizer.num_special_tokens_to_add(False)
        windows, groups = [], []
        for text in texts:
            if "multilingual-e5" in self.model_name:
                text = f"{role}: {text}"
            ids = tokenizer.encode(text, add_special_tokens=False)
            start = len(windows)
            weights = []
            for offset in range(0, max(1, len(ids)), width):
                part = ids[offset:offset + width]
                windows.append(tokenizer.decode(part, skip_special_tokens=True))
                weights.append(max(1, len(part)))
            groups.append((start, len(windows), weights))
        if not windows:
            return np.empty((0, self.signature["dimension"]), dtype=np.float32)
        encoded = self.model.encode(windows, convert_to_numpy=True, show_progress_bar=False, batch_size=16)
        return normalize(np.stack([np.average(encoded[a:b], axis=0, weights=w) for a, b, w in groups]))

    def encode_query(self, texts):
        return self.encode(texts, role="query")
