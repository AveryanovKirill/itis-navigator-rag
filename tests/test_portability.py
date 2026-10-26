"""Проверки первого запуска без исходных отчётов и закрытого корпуса."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path
import httpx
import pytest


def test_mcp_check_explicit_question_does_not_read_evaluation_set(monkeypatch):
    import src.cli as cli
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def fake_session():yield object()
    import src.mcp_client as mc
    monkeypatch.setattr(mc,"connect_mcp",fake_session)
    async def search(session,q):return {"status":"not_found","fragments":[],"query":q}
    monkeypatch.setattr(mc,"call_search",search)
    monkeypatch.setattr(cli,"read_json",lambda p: (_ for _ in ()).throw(AssertionError("Нельзя читать эталоны при явном запросе")))
    monkeypatch.setattr(cli,"save_json",lambda *a:None)
    asyncio.run(cli.mcp_demo("Любой локальный запрос"))


def test_missing_index_fails_before_embedding_download(tmp_path,monkeypatch):
    import src.search as search
    monkeypatch.setattr(search,"LocalEmbeddings",lambda: (_ for _ in ()).throw(AssertionError("Не нужно загружать модель")))
    with pytest.raises(ValueError,match="src.bootstrap"):
        search.load_search(tmp_path,tmp_path)


@pytest.mark.parametrize("status,models,expected",[(200,[],"ollama pull qwen3:4b"),(503,[],"Ollama недоступна")])
def test_doctor_reports_missing_server_or_model(monkeypatch,status,models,expected):
    import src.bootstrap as bootstrap
    real=httpx.AsyncClient
    def handle(request):return httpx.Response(status,json={"version":"TEST","models":models})
    monkeypatch.setattr(bootstrap.httpx,"AsyncClient",lambda **kw:real(transport=httpx.MockTransport(handle),**kw))
    with pytest.raises(ValueError,match=expected):asyncio.run(bootstrap.check_ollama())


def test_ollama_custom_base_url_and_fixed_qwen(monkeypatch):
    import src.agent as agent
    real=httpx.AsyncClient
    def handle(request):
        assert str(request.url)=="http://localhost:12345/api/chat"
        assert json.loads(request.content)["model"]=="qwen3:4b"
        return httpx.Response(200,json={"model":"qwen3:4b","message":{"content":"{}"},"prompt_eval_count":1,"eval_count":1})
    monkeypatch.setattr(agent,"OLLAMA_BASE_URL","http://localhost:12345")
    monkeypatch.setattr(agent.httpx,"AsyncClient",lambda **kw:real(transport=httpx.MockTransport(handle),**kw))
    asyncio.run(agent.Ollama().complete([{"role":"user","content":"test"}]))


def test_bootstrap_keeps_complete_index_without_rebuild(tmp_path,monkeypatch):
    import src.bootstrap as bootstrap
    import src.embeddings as embeddings
    import src.indexing as indexing
    import src.search as search
    monkeypatch.setattr(bootstrap,"ROOT",tmp_path)
    (tmp_path/"docs").mkdir();(tmp_path/"docs/a.txt").write_text("Разрешённый текст.",encoding="utf-8")
    (tmp_path/"index").mkdir()
    for name in ("chunks.json","metadata.json","embeddings.npy"):(tmp_path/"index"/name).write_bytes(b"KEEP")
    monkeypatch.setattr(embeddings,"LocalEmbeddings",lambda:object())
    monkeypatch.setattr(indexing,"VectorIndex",lambda *a,**kw:object())
    monkeypatch.setattr(search,"relevance_threshold",lambda i:.8)
    monkeypatch.setattr(indexing,"build_index",lambda *a,**kw: (_ for _ in ()).throw(AssertionError("Не перестраивать")))
    bootstrap.prepare()
    assert all(p.read_bytes()==b"KEEP" for p in (tmp_path/"index").iterdir())


def test_schedule_absent_dependency_does_not_break_document_server(monkeypatch,tmp_path):
    import src.schedule as schedule
    monkeypatch.setattr(schedule,"ROOT",tmp_path)
    result=asyncio.run(schedule.get_group_schedule("11-307","понедельник"))
    assert result["status"]=="unavailable" and "setup_schedule" in result["message"]


def test_optional_education_page_failure_keeps_five_imported_documents(monkeypatch,tmp_path):
    import src.official as official
    import pypdf
    from types import SimpleNamespace
    monkeypatch.setattr(official,"ROOT",tmp_path)
    text="Тестовый текст для проверки импортёра. "*20
    def fetch(url):
        if url==official.EDUCATION:raise ValueError("Тестовая недоступность необязательного ресурса")
        return SimpleNamespace(content=b"TEST_ONLY",text=f'<main><h1>Тест</h1><p>{text}</p></main>')
    monkeypatch.setattr(official,"fetch",fetch)
    monkeypatch.setattr(pypdf,"PdfReader",lambda *a:SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda:text)]))
    official.import_official()
    assert len(list((tmp_path/"docs").glob("*.txt")))==5
    assert (tmp_path/"data/sources.json").is_file()
