import json
import numpy as np
import pytest

from src.embeddings import normalize, cosine_matrix
from src.chunking import split_text, make_chunks, read_documents
from src.indexing import build_index, VectorIndex
from src.storage import save_json, read_json
from src.search import search_payload
from src.exercises import gold_ids, validate_dataset


def test_normalization_cosine():
    vectors = np.array([[3, 4], [0, 0], [-3, -4]], dtype=float)
    assert np.allclose(normalize(vectors), [[.6, .8], [0, 0], [-.6, -.8]])
    assert np.allclose(cosine_matrix(vectors, vectors), [[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
    assert np.allclose(vectors[0], [3, 4])
    with pytest.raises(ValueError):
        normalize([[np.nan, 1]])


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)])
def test_bad_chunk_options(size, overlap):
    with pytest.raises(ValueError):
        split_text("", "test.txt", size, overlap)


def test_chunk_edges():
    assert split_text(" \n", "a.txt") == []
    assert split_text("короткий", "a.txt")[0]["text"] == "короткий"
    text = "0123456789" * 3 + "Z"
    chunks = split_text(text, "a.txt", 10, 3)
    assert [c["start"] for c in chunks] == [0, 7, 14, 21]
    assert chunks[-1]["text"] == text[21:31]
    for a, b in zip(chunks, chunks[1:]):
        assert a["text"][-3:] == b["text"][:3]
    assert len(split_text("x" * 11, "b.txt", 10, 3)[-1]["text"]) == 4
    assert len(split_text("x" * 10, "b.txt", 10, 3)) == 1


def test_reading_errors(tmp_path):
    with pytest.raises(ValueError, match="нет"):
        read_documents(tmp_path)
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x80")
    with pytest.raises(ValueError, match="UTF-8"):
        read_documents(tmp_path)


def make_test_index(tmp_path, toy):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("кот кот", encoding="utf-8")
    (docs / "b.md").write_text("пёс пёс", encoding="utf-8")
    (docs / "c.txt").write_text("чай чай", encoding="utf-8")
    directory = tmp_path / "index"
    chunks, _ = make_chunks(docs, directory / "chunks.json")
    build_index(directory / "chunks.json", directory, toy, docs)
    return VectorIndex(directory, toy, docs), docs, directory


def test_index_topk_persist_threshold(tmp_path, toy):
    index, docs, directory = make_test_index(tmp_path, toy)
    assert index.search("кот", k=1)[0]["source"] == "a.txt"
    assert len(index.search("кот")) == 3
    assert len(index.search("кот", threshold=.9)) == 1
    assert search_payload(index, "рыба", .3)["status"] == "not_found"
    assert index.search(" ") == []
    with pytest.raises(ValueError):
        index.search("кот", k=4)
    with pytest.raises(ValueError):
        index.search("кот", threshold=1.1)
    # Проверяем, что поиск не перекодирует документы.
    calls = []
    old_encode = toy.encode
    toy.encode = lambda texts: (calls.append(texts) or old_encode(texts))
    index.search("кот")
    assert calls == [["кот"]]
    (docs / "a.txt").write_text("кот новый", encoding="utf-8")
    with pytest.raises(ValueError, match="изменены"):
        VectorIndex(directory, toy, docs)


def test_corrupt_or_wrong_model_index(tmp_path, toy):
    index, docs, directory = make_test_index(tmp_path, toy)
    metadata = read_json(directory / "metadata.json")
    metadata["embedding"] = {"model": "other"}
    save_json(directory / "metadata.json", metadata)
    with pytest.raises(ValueError, match="отличаются"):
        VectorIndex(directory, toy, docs)


def test_empty_index_build(tmp_path, toy):
    save_json(tmp_path / "chunks.json", [])
    with pytest.raises(ValueError, match="Нет фрагментов"):
        build_index(tmp_path / "chunks.json", tmp_path, toy, tmp_path)


def test_plain_index_mode_for_reproducible_baseline(tmp_path, toy):
    _, docs, directory = make_test_index(tmp_path, toy)
    calls = []
    old = toy.encode
    toy.encode = lambda texts: (calls.extend(texts) or old(texts))
    metadata = build_index(directory / "chunks.json", directory, toy, docs, contextual_title=False)
    assert metadata["text_template"] == "plain_chunk_v1"
    assert calls == ["кот кот", "пёс пёс", "чай чай"]


def test_real_dataset_has_independent_evidence():
    from src.config import ROOT
    questions = read_json(ROOT / "eval/questions_50.json")
    chunks = read_json(ROOT / "index/chunks.json")
    gold = validate_dataset(questions, chunks)
    assert len(gold) == 50
    assert all(gold[q["id"]] for q in questions if q["answerable"])
    sources = read_json(ROOT / "data/sources.json")
    for q in questions:
        if q["answerable"]:
            assert q["source_url"] == sources[q["source"]]["url"]
    broken = dict(questions[0], evidence="Такого предложения в документе нет.")
    with pytest.raises(ValueError):
        gold_ids(broken, chunks)
