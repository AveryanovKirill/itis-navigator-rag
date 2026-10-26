import csv
import numpy as np

from .config import ROOT
from .storage import save_json, read_json, fingerprint

E1_PHRASES = ["Как оформить справку о том, что я учусь?", "Где заказать документ, подтверждающий моё обучение?",
              "Из-за болезни я пропустил экзамен.", "Я не смог прийти на экзамен, потому что заболел.",
              "Студент разрешил преподавателю перенести экзамен.", "Преподаватель разрешил студенту перенести экзамен."]


def e1(embedder):
    expected = {"phrases": E1_PHRASES, "pairs": [[1, 2], [3, 4], [5, 6]],
                "prediction": "Ожидаю высокую близость 1–2 и 3–4; 5–6 имеют общие слова, но меняют роли. Хотелось бы similarity(5,6) ниже обеих смысловых пар, однако модель может не различить роли."}
    # Файл записывается ДО вычисления эмбеддингов.
    save_json(ROOT / "reports/e1_expectations.json", expected)
    vectors = embedder.encode(E1_PHRASES)
    matrix = vectors @ vectors.T
    path = ROOT / "reports/e1_similarity.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Фраза"] + list(range(1, 7)))
        for phrase, row in zip(E1_PHRASES, matrix):
            writer.writerow([phrase] + [f"{v:.6f}" for v in row])
    pairs = [float(matrix[i, j]) for i, j in [(0, 1), (2, 3), (4, 5)]]
    matched = pairs[2] < min(pairs[:2])
    explanation = ("Ожидаемый порядок выполнен." if matched else "Ожидаемый порядок не выполнен: модель чувствительна к общим словам и может путать роли участников.")
    explanation += " Косинусная близость не доказывает логическую эквивалентность. Смена субъекта и объекта — сложный случай для компактной модели."
    result = {"embedding": embedder.signature, "matrix": matrix.tolist(), "pair_scores": pairs,
              "expected_order_met": matched, "explanation": explanation}
    save_json(ROOT / "reports/e1_results.json", result)
    print("        " + " ".join(f"{i:>7}" for i in range(1, 7)))
    for i, row in enumerate(matrix):
        print(f"{i+1:>7} " + " ".join(f"{v:7.3f}" for v in row))
    for i, phrase in enumerate(E1_PHRASES, 1):
        print(f"{i}. {phrase}")
    print(explanation)
    return result


def gold_ids(item, chunks):
    if not item.get("answerable", True):
        return []
    evidence, source = item["evidence"], item["source"]
    ids = [c["id"] for c in chunks if c["source"] == source and evidence in c["text"]]
    if not ids:
        raise ValueError(f"Цитата {item['id']} не найдена целиком в chunks; проверьте корпус/эталоны. Эталоны не подгоняются под выдачу.")
    return ids


def validate_dataset(items, chunks):
    if len(items) != 50 or len({x["question"] for x in items}) != 50 or len({x["id"] for x in items}) != 50:
        raise ValueError("Нужны ровно 50 различных вопросов и ID.")
    if sum(x["answerable"] for x in items) != 40:
        raise ValueError("Нужны 40 answerable и 10 unanswerable.")
    return {q["id"]: gold_ids(q, chunks) for q in items}


def recall(index):
    questions = read_json(ROOT / "eval/retrieval_15.json")
    if len(questions) != 15:
        raise ValueError("Нужно 15 retrieval-вопросов.")
    # Сначала фиксируем ВСЕ эталоны по цитатам, только потом поиск.
    gold = {q["id"]: gold_ids(q, index.chunks) for q in questions}
    save_json(ROOT / "reports/retrieval_gold.json", gold)
    rows = []
    for q in questions:
        hits = index.search(q["question"])
        success = bool(set(gold[q["id"]]) & {h["id"] for h in hits})
        rows.append(dict(q, relevant_chunks=gold[q["id"]], hits=hits, hit=success))
    count = sum(r["hit"] for r in rows)
    result = {"index_hash": fingerprint(index.metadata), "hits": count, "total": 15,
              "recall_at_3": count / 15, "definition": "Доля вопросов с хотя бы одним эталонным chunk в top-3 (hit rate по условию задания)",
              "failed_ids": [r["id"] for r in rows if not r["hit"]], "details": rows}
    save_json(ROOT / "reports/recall.json", result)
    print(f"Recall@3 = {count}/15 = {count/15:.3f}. Цель >= 12/15. Неудачи: {result['failed_ids']}")
    return result


def calibrate(index):
    rows = []
    for q in read_json(ROOT / "eval/calibration.json"):
        hits = index.search(q["question"], k=1)
        rows.append(dict(q, score=hits[0]["score"] if hits else -1))
    values = sorted(set(r["score"] for r in rows))
    candidates = [-1.0] + [(a+b)/2 for a, b in zip(values, values[1:])] + [1.0]
    trials = []
    for t in candidates:
        positives = [r for r in rows if r["answerable"]]
        negatives = [r for r in rows if not r["answerable"]]
        tp = sum(r["score"] >= t for r in positives)
        tn = sum(r["score"] < t for r in negatives)
        trials.append({"threshold": t, "balanced_accuracy": (tp/len(positives)+tn/len(negatives))/2,
                       "positive_recall": tp/len(positives), "negative_rejection": tn/len(negatives),
                       "true_positive": tp, "true_negative": tn})
    # При равенстве качества предпочитаем меньше ложных допусков.
    feasible = [r for r in trials if r["positive_recall"] >= .9]
    best = max(feasible, key=lambda r: (r["balanced_accuracy"], r["true_negative"], r["threshold"]))
    result = dict(best, index_hash=fingerprint(index.metadata), dataset_hash=fingerprint(read_json(ROOT / "eval/calibration.json")),
                  method="Максимум balanced accuracy при positive recall >= 0.9 на отдельном наборе; при равенстве — более строгий порог. Ограничение уменьшает ложные отказы retrieval, семантическую достаточность проверяет модель и человек.",
                  limitation="Это калибровка, а не независимая оценка. Тематически близкие вопросы без ответа могут пройти порог. Отсутствие ответа проверяет также LLM и человек.",
                  details=rows, trials=trials)
    save_json(ROOT / "reports/threshold.json", result)
    print(f"Порог: {best['threshold']:.4f}; balanced accuracy калибровки: {best['balanced_accuracy']:.3f}.")
    return result
