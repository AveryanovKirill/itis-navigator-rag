"""Отображение уже выбранных моделью номеров sources, без изменения фактов."""
import re
from .schemas import Answer


def render_single_sentence_sources(raw, fragments):
    value = Answer.model_validate_json(raw)
    if not value.found or not value.sources or re.search(r"\[\d+\]", value.answer):
        return raw, None
    registry = {f["number"] for f in fragments}
    if not set(value.sources) <= registry:
        return raw, None  # Не добавляем и не исправляем придуманные номера.
    protected = re.sub(r"\b(г|ул|д|ауд|п|им|т|доб|кв|стр|руб)\.", r"\1<DOT>", value.answer, flags=re.I)
    protected = re.sub(r"\b([А-ЯЁ])\.", r"\1<DOT>", protected)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", protected) if s.strip()]
    if len(sentences) != 1 or value.answer.rstrip().endswith("?"):
        return raw, None  # Для нескольких утверждений ссылки должна расставить сама LLM.
    original = value.answer
    body = original.rstrip()
    ending = body[-1] if body[-1] in ".!" else "."
    body = body[:-1].rstrip() if body[-1] in ".!" else body
    value.answer = body + " " + " ".join(f"[{n}]" for n in value.sources) + ending
    return value.model_dump_json(), {"kind": "render_existing_sources_for_single_sentence",
                                     "original_text": original, "display_text": value.answer,
                                     "sources_unchanged": value.sources}
