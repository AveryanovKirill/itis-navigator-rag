"""Проверяемое извлечение HTML КФУ без выполнения JavaScript."""
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
import re
import hashlib
import httpx
from .config import ROOT
from .storage import save_json, now


class Node:
    def __init__(self, tag="root", attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()

    def text(self, base="", links=False):
        if self.tag in {"script", "style", "noscript", "nav", "header", "footer", "form"}:
            return ""
        value = "".join(c.text(base, links) if isinstance(c, Node) else c for c in self.children)
        if links and self.tag == "a" and self.attrs.get("href") and value.strip():
            url = urljoin(base, self.attrs["href"])
            if url.startswith("https://"):
                value += " (" + url + ")"
        if self.tag in {"p", "div", "li", "h1", "h2", "h3", "tr", "br"}:
            value = "\n" + value + "\n"
        if self.tag in {"td", "th"}:
            value += " | "
        return value


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in {"br", "img", "meta", "link", "input", "hr", "source", "wbr", "area", "base", "embed", "param", "col"}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack = self.stack[:i]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def extract_html(html, url):
    parser = PageParser()
    parser.feed(html)
    nodes = list(parser.root.walk())
    roots = [n for n in nodes if n.attrs.get("id") == "ss_content"]
    if not roots:
        roots = [n for n in nodes if n.tag in {"article", "main"}]
    if not roots:
        raise ValueError("Не найдена область основного текста; требуется ручная проверка структуры страницы.")
    root = max(roots, key=lambda n: len(n.text()))
    text = root.text(url, links=True).replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    links = [{"title": re.sub(r"\s+", " ", n.text()).strip(), "url": urljoin(url, n.attrs["href"])}
             for n in root.walk() if n.tag == "a" and n.attrs.get("href")]
    return text, links


def fetch(url):
    host = urlparse(url).hostname or ""
    if urlparse(url).scheme != "https" or not (host == "kpfu.ru" or host.endswith(".kpfu.ru")):
        raise ValueError("Для документов разрешён официальный HTTPS-домен КФУ.")
    for attempt in range(2):
        try:
            with httpx.Client(timeout=25, follow_redirects=False) as client:
                response = client.get(url)
                response.raise_for_status()
            break
        except httpx.HTTPError as exc:
            if attempt == 1:
                raise ValueError(f"Не удалось скачать официальный источник {url}. Существующие docs/ не заменены. Можно вручную сохранить UTF-8 текст и обновить data/sources.json. Причина: {type(exc).__name__}") from exc
    if "charset=windows-1251" in response.headers.get("content-type", "").lower():
        response.encoding = "windows-1251"
    return response


PAGES = {
    "01_itis.txt": "https://kpfu.ru/itis",
    "02_scholarships.txt": "https://kpfu.ru/itis/obuchenie-v-itis/stipendialnye-programmy",
    "03_dormitory.txt": "https://kpfu.ru/itis/zaselenie-v-obschezhitiya-itis",
    "04_practice.txt": "https://kpfu.ru/itis/obuchenie-v-itis/praktika",
}
REGULATION = "https://kpfu.ru/portal/docs/F_626640358/Reg..tek..kontrolya.s.DOT.pdf"
EDUCATION = "https://kpfu.ru/itis/obuchenie-v-itis"


def import_official(reuse_downloads=False):
    """Снимки по пяти источникам. Сначала скачиваем, потом обновляем docs/.

    Исходные байты, URL, время и SHA256 сохраняются в data/official/.
    Это не утверждение юридической актуальности регламента 2020 года.
    """
    from pypdf import PdfReader
    from io import BytesIO
    records, texts = {}, {}
    (ROOT / "data/official").mkdir(parents=True, exist_ok=True)
    # Возобновление первой загрузки: используем только файлы с проверяемой
    # привязкой URL и SHA256; голые/чужие файлы автоматически не принимаются.
    cache_manifest = ROOT / "data/official/downloads.json"
    import json
    cached = json.loads(cache_manifest.read_text(encoding="utf-8")) if reuse_downloads and cache_manifest.exists() else {}
    def download(url, path):
        key = path.name
        prior = cached.get(key, {})
        if reuse_downloads and path.exists() and prior.get("url") == url and hashlib.sha256(path.read_bytes()).hexdigest() == prior.get("sha256"):
            print("Повторно используется проверенная локальная загрузка: " + key)
            return httpx.Response(200, content=path.read_bytes(), headers={"Content-Type": prior.get("content_type", "")}), prior["fetched_at"]
        response = fetch(url)
        path.write_bytes(response.content)
        stamp = now()
        cached[key] = {"url": url, "sha256": hashlib.sha256(response.content).hexdigest(), "fetched_at": stamp,
                       "content_type": response.headers.get("content-type", "") if hasattr(response, "headers") else ""}
        save_json(cache_manifest, cached)
        return response, stamp
    for filename, url in PAGES.items():
        raw_path = ROOT / "data/official" / filename.replace(".txt", ".html")
        response, stamp = download(url, raw_path)
        content, links = extract_html(response.text, url)
        # На главной исключаем динамическую ленту новостей после описания/контактов.
        if filename == "01_itis.txt":
            content = content.split("Фотоотчеты")[0]
        records[filename] = {"url": url, "fetched_at": stamp, "sha256": hashlib.sha256(response.content).hexdigest(),
                             "raw_file": raw_path.relative_to(ROOT).as_posix(), "links": links,
                             "validity": "Содержимое официальной страницы на момент получения; дата загрузки не гарантирует актуальность каждого положения."}
        texts[filename] = content
    response, stamp = download(REGULATION, ROOT / "data/official/assessment.pdf")
    reader = PdfReader(BytesIO(response.content))
    texts["05_assessment.txt"] = "\n\n".join(f"Страница PDF {i+1}\n" + (p.extract_text() or "").replace("\x02", "") for i, p in enumerate(reader.pages))
    records["05_assessment.txt"] = {"url": REGULATION, "fetched_at": stamp, "sha256": hashlib.sha256(response.content).hexdigest(),
                                   "raw_file": "data/official/assessment.pdf", "pages": len(reader.pages),
                                   "validity": "Регламент ДОТ 2020 года, ссылка опубликована на странице ИТИС. Действие до отмены; отсутствие отмены не проверено. Не переносить специальные нормы ДОТ на любую очную аттестацию."}
    # Страница расписания не является частью пяти документов основного RAG.
    # Её недоступность не должна отменять уже успешную загрузку корпуса.
    try:
        response = fetch(EDUCATION)
        content, links = extract_html(response.text, EDUCATION)
        save_json(ROOT / "data/education_page.json", {"url": EDUCATION, "fetched_at": now(), "text": content, "links": links})
    except ValueError:
        print("Необязательная страница расписания недоступна; основной корпус документов будет сохранён.")
    for filename, text in texts.items():
        if len(text) < 300:
            raise ValueError(f"Недостаточно извлечённого текста: {filename}.")
    (ROOT / "docs").mkdir(exist_ok=True)
    for filename, text in texts.items():
        (ROOT / "docs" / filename).write_text(text + "\n", encoding="utf-8")
    save_json(ROOT / "data/sources.json", records)
    print("Сохранены 5 официальных источников КФУ, исходные файлы и метаданные.")


if __name__ == "__main__":
    import_official()
