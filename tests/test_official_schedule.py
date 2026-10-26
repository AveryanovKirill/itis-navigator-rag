import asyncio
from src.official import extract_html
from src.schedule import schedule_args, schedule_intent, get_group_schedule


def test_official_extractor_removes_commands_and_navigation():
    html = '<html><nav>Навигация</nav><div id="content"></div><div id="ss_content"><h1>Практика</h1><p>Документ</p><script>evil()</script><a href="/itis/file">Форма</a></div></html>'
    text, links = extract_html(html, "https://kpfu.ru/itis")
    assert "Практика" in text and "Документ" in text
    assert "evil()" not in text and "Навигация" not in text
    assert links[0]["url"] == "https://kpfu.ru/itis/file"


def test_schedule_requires_group_and_routes_separately():
    assert schedule_intent("Какие у меня пары завтра?")
    assert schedule_intent("Во сколько первая пара 11-307?")
    assert not schedule_intent("Когда пересдача у группы 11-307?")
    assert not schedule_intent("В какой аудитории консультируют по стипендиям?")
    assert schedule_args("Пары группы 11-307 в понедельник") == ("11-307", "понедельник")
    assert schedule_args("Пары завтра", "11-307") == ("11-307", "завтра")
    assert schedule_args("Пары в среду", "11-307") == ("11-307", "среда")
    assert asyncio.run(get_group_schedule())["status"] == "need_group"
