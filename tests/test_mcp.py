import asyncio
import pytest
from src.mcp_client import connect_mcp, call_search
from .test_core import make_test_index


@pytest.mark.integration
def test_real_stdio_protocol_with_test_vectors(tmp_path, toy):
    _, docs, directory = make_test_index(tmp_path, toy)
    async def run():
        async with connect_mcp(["-m", "tests.mcp_fixture", str(directory), str(docs)]) as session:
            tools = await session.list_tools()
            assert "search_docs" in {t.name for t in tools.tools}
            found = await call_search(session, "кот")
            assert found["status"] == "found"
            assert found["fragments"][0]["source"] == "a.txt"
            assert len(found["fragments"]) == 1
            empty = await call_search(session, "рыба")
            assert empty["status"] == "not_found"
            bad = await session.call_tool("search_docs", {"query": ""})
            assert bad.isError
    asyncio.run(run())




@pytest.mark.local_model
def test_actual_vectors_and_long_input():
    import numpy as np
    from src.embeddings import LocalEmbeddings
    embedder = LocalEmbeddings()
    vectors = embedder.encode(["Студент получает стипендию.", "Учащемуся выплачивают стипендию.", "кот " * 400 + "практика в ИТИС"])
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert float(vectors[0] @ vectors[1]) > .5
