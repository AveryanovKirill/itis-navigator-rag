import re
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator

REFUSAL = "В доступных документах недостаточно информации для ответа."


class Answer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    answer: Annotated[str, Field(min_length=1)]
    sources: list[Annotated[int, Field(ge=1)]]
    found: bool

    @model_validator(mode="after")
    def check_state(self):
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("Номера источников не должны повторяться.")
        if not self.found and (self.sources or self.answer != REFUSAL):
            raise ValueError("При found=false допускается только стандартный отказ без источников и фактов.")
        return self


def validate_answer(raw, fragments, version="v2"):
    answer = Answer.model_validate_json(raw)
    registry = {f["number"]: f for f in fragments}
    if not set(answer.sources) <= set(registry):
        raise ValueError("Ответ ссылается на несуществующий источник.")
    urls = {u.rstrip('.,;') for u in re.findall(r"https?://[^\s)\]>]+", answer.answer)}
    if not urls <= {registry[n].get('url') for n in answer.sources}:
        raise ValueError("В ответе есть URL, отсутствующий в использованных источниках.")
    markers = set(map(int, re.findall(r"\[(\d+)\]", answer.answer)))
    if markers != set(answer.sources):
        raise ValueError("Маркеры [n] в тексте должны совпадать с sources.")
    if version == "v2" and answer.found:
        if not fragments or not answer.sources:
            raise ValueError("Без найденных источников нельзя возвращать фактический ответ.")
        # Проверка чисел по каждому предложению и его собственным ссылкам.
        # Это консервативная эвристика, не семантическое доказательство.
        protected = re.sub(r"\b(г|ул|д|ауд|п|им|т|доб|кв|стр|руб)\.", r"\1<DOT>", answer.answer, flags=re.I)
        protected = re.sub(r"\b([А-ЯЁ])\.", r"\1<DOT>", protected)
        sentences = re.split(r"(?<=[.!?])\s+|\n+", protected)
        for sentence in sentences:
            if not sentence.strip():
                continue
            refs = set(map(int, re.findall(r"\[(\d+)\]", sentence)))
            if not refs:
                raise ValueError("Каждое предложение фактического ответа должно содержать [n] до точки.")
            clean = re.sub(r"\[\d+\]", "", sentence)
            cited = " ".join(registry[n]["text"] for n in refs)
            if not set(re.findall(r"\d+", clean)) <= set(re.findall(r"\d+", cited)):
                raise ValueError("В ответе есть число/дата, отсутствующее в указанных источниках.")
    return answer


def fallback():
    return Answer(answer=REFUSAL, sources=[], found=False)
