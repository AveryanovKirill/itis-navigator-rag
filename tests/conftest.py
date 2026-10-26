import pytest
import numpy as np


class ToyEmbeddings:
    """Только unit-тесты алгоритма; не используется для E1/E3 или отчётов."""
    signature = {"model": "TEST_ONLY", "dimension": 3}

    def encode(self, texts):
        return np.array([[float("кот" in t), float("пёс" in t), float("чай" in t)] for t in texts], dtype=np.float32)


@pytest.fixture
def toy():
    return ToyEmbeddings()


def pytest_addoption(parser):
    parser.addoption("--run-local-model", action="store_true", help="Проверить настоящую локальную модель и production MCP, нужна загруженная модель")


def pytest_configure(config):
    # pytest --basetemp не создаёт отсутствующий родительский каталог.
    # Тесты должны запускаться сразу после clone, ещё до скачивания моделей.
    (config.rootpath / ".cache").mkdir(exist_ok=True)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-local-model"):
        for item in items:
            if "local_model" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Добавьте --run-local-model для настоящих эмбеддингов"))
