import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Кэши и загрузки остаются внутри проекта, включая процессы MCP.
for name, value in {
    "HF_HOME": ROOT / ".cache/huggingface",
    "TORCH_HOME": ROOT / ".cache/torch",
    "XDG_CACHE_HOME": ROOT / ".cache",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_DISABLE_XET": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUTF8": "1",
}.items():
    os.environ[name] = str(value)

from dotenv import load_dotenv

if not os.getenv("ITIS_MCP_SERVER"):
    load_dotenv(ROOT / ".env")
MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")


def api_key():
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key or key == "YOUR_OPENROUTER_KEY":
        raise ValueError("API-ключ отсутствует. Скопируйте .env.example в .env и заполните OPENROUTER_API_KEY. E1–E3 и mcp-check работают без ключа.")
    return key
