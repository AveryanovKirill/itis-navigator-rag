import json
import pytest
from src.schemas import Answer, validate_answer, fallback, REFUSAL

HITS = [{"number": 1, "text": "Заселение 28 августа 2026 года."}, {"number": 2, "text": "Время 10:00."}]


def raw(answer="Заселение 28 августа 2026 года [1].", sources=None, found=True):
    return json.dumps({"answer": answer, "sources": [1] if sources is None else sources, "found": found})


def test_valid_answer_and_refusal():
    assert validate_answer(raw(), HITS).sources == [1]
    assert validate_answer(fallback().model_dump_json(), []).found is False
    assert validate_answer(raw("Общий ответ", [], True), [], "v1").found


@pytest.mark.parametrize("data", ["not json", '{"answer":"x","sources":["1"],"found":true}',
    '{"answer":"x","sources":[],"found":"false"}', '{"answer":"x","sources":[],"found":true,"extra":1}',
    '{"answer":"x","sources":[true],"found":true}'])
def test_strict_json(data):
    with pytest.raises(ValueError):
        Answer.model_validate_json(data)


def test_nonexistent_sources_and_false_refusal_date():
    with pytest.raises(ValueError):
        validate_answer(raw("Экзамен 20 мая [3].", [3]), HITS)
    with pytest.raises(ValueError):
        validate_answer(raw("Экзамен 20 мая", [], False), [])
    with pytest.raises(ValueError):
        validate_answer(raw("Заселение 29 августа [1]."), HITS)
    with pytest.raises(ValueError):
        validate_answer(raw(), [])
    with pytest.raises(ValueError):
        validate_answer(raw("Заселение 28 августа 2026 года.", [1]), HITS)
    with pytest.raises(ValueError):
        validate_answer(raw("Заселение 28 августа 2026 года [1]. А ещё есть правило.", [1]), HITS)


def test_validator_is_not_semantic_proof():
    # Номера существуют, но отношение выдумано. Это специально документирует границу проверки.
    assert validate_answer(raw("Ректор живёт в общежитии [1]."), HITS).found


def test_address_abbreviations_do_not_break_citations():
    assert validate_answer(raw("Адрес: г. Казань, ул. Кремлевская, д. 28 [1]."), HITS).found


def test_fabricated_url_rejected():
    with pytest.raises(ValueError, match="URL"):
        validate_answer(raw("Смотрите https://invented.invalid/ [1]."), HITS)
