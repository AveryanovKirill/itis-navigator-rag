"""Отдельная интеграция расписания с рабочим MCP-Demo. Не участвует в E3/E5."""
import json
import os
import re
import shutil
from datetime import timedelta
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .config import ROOT
from .storage import read_json, fingerprint, save_json, now

OFFICIAL_PAGE = "https://kpfu.ru/itis/obuchenie-v-itis"
OFFICIAL_YEAR4 = "https://disk.360.yandex.ru/i/G0ejiEXQxBoSFA"


def schedule_intent(question):
    if re.search(r"пересдач|задолженн|комисси", question, re.I):
        return False
    return bool(re.search(r"расписан|\bпары\b|\bпар[ауе]\b|занят[иь]|кто вед[её]т предмет", question, re.I))


def schedule_args(question, group=None):
    match = re.search(r"11(?:\.1)?[-–—]\d{3}[а-яa-z]?", question, re.I)
    group = group or (match.group().replace("–", "-").replace("—", "-") if match else "")
    days = {"понедельник": "понедельник", "вторник": "вторник", "сред": "среда", "четверг": "четверг", "пятниц": "пятница", "суббот": "суббота", "воскресен": "воскресенье", "сегодня": "сегодня", "послезавтра": "послезавтра", "завтра": "завтра"}
    day = next((v for k, v in days.items() if k in question.lower()), "")
    date = re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
    return group, date.group() if date else day


@asynccontextmanager
async def reference_mcp():
    server = ROOT / "reference/MCP-Demo/server/mcp-server.mjs"
    if not server.exists() or not shutil.which("node"):
        raise ValueError("Для расписания выполните scripts/setup_schedule.ps1 (нужны Node.js >=22.13 и Git). Основной RAG работает независимо.")
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    params = StdioServerParameters(command=shutil.which("node"), args=[str(server)], cwd=str(server.parents[1]), env=env)
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=150)) as session:
            await session.initialize()
            tools = await session.list_tools()
            if "get_group_schedule" not in {t.name for t in tools.tools}:
                raise ValueError("Исходный сервер не объявил get_group_schedule.")
            yield session


def lesson_signature(lessons):
    return fingerprint([{k: re.sub(r"\s+", " ", h[k]).strip() for k in ["day", "time", "details"]} for h in lessons])


async def get_group_schedule(group="", day=""):
    if not group:
        return {"status": "need_group", "tool": "get_group_schedule", "message": "Укажите группу ИТИС, например 11-307. Группа не угадывается.",
                "official_page": OFFICIAL_PAGE, "lessons": []}
    try:
        async with reference_mcp() as session:
            response = await session.call_tool("get_group_schedule", {"group": group, "day": day, "refresh": True})
            data = json.loads("\n".join(c.text for c in response.content if c.type == "text"))
            if response.isError:
                raise ValueError(data.get("error", "Ошибка исходного инструмента расписания."))
    except (ValueError, OSError) as exc:
        return {"status": "unavailable", "tool": "get_group_schedule", "message": str(exc), "lessons": [], "official_page": OFFICIAL_PAGE}
    data["tool"] = "get_group_schedule"
    data["data_kind"] = "schedule_separate_from_rag"
    data["official_page"] = OFFICIAL_PAGE
    data["official_pdf_year4"] = OFFICIAL_YEAR4
    data["status"] = "found" if data.get("lessons") else "no_entries"
    verification_path = ROOT / "data/schedule/verification.json"
    checked = read_json(verification_path) if verification_path.exists() else {}
    signature = lesson_signature(data.get("lessons", []))
    data["verified_against_official_pdf"] = bool(checked.get("group") == group and checked.get("day") == data.get("day") and checked.get("lessons_signature") == signature)
    data["provenance_note"] = "Публичная Google-таблица MCP-Demo; официальный сайт ИТИС публикует PDF. Проверенные ячейки 11-307/понедельник сверены визуально с PDF 19.10.2026. Для остальных ячеек совпадение автоматически не установлено. Условия недель/подгрупп и отмены не вычисляются."
    if data.get("mode") != "live":
        data["status"] = "stale_snapshot"
    return data


async def check_schedule(group="11-307", day="понедельник"):
    data = await get_group_schedule(group, day)
    save_json(ROOT / "reports/schedule_mcp_check.json", {"timestamp": now(), "real_mcp": True, "data": data, "gemini_called": False})
    return data
