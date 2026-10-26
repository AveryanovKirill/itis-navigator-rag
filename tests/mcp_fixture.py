"""Сервер ТОЛЬКО для интеграционного теста транспорта с тестовыми векторами."""
import sys
from pathlib import Path
from tests.conftest import ToyEmbeddings
from src.indexing import VectorIndex
from src.mcp_server import create_server

if __name__ == "__main__":
    index = VectorIndex(Path(sys.argv[1]), ToyEmbeddings(), Path(sys.argv[2]))
    create_server(lambda: (index, .6)).run(transport="stdio")
