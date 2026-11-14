from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field 

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient


# api/.env
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        prefix="RAG_",
        env_file=ENV_FILE,
        extra="ignore"
    )

    # App
    app_name: str = "Clause"
    data_path: str = "files/shop_handbook.pdf"

    # Chat
    chat_provider: str = "ollama"
    chat_model: str = "gemma3:1b"
    chat_temperature: float = 0.1
    chat_api_key: str = ""

    # Embed
    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text:latest"
    embed_dim: int = 768

    # Indexing
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Retrieval
    top_k: int = 3

    # Qdrant
    collection: str = "shop_handbook"  # table
    qdrant_data: str = "./vector_store" # vector store
    # qdrant_url

    # Prompts
    system_prompt: str = (
        "You are the Clause shop customer assistant. "
        "Answer using ONLY the handbook text you are given. "
        "Find the one sentence that answers the question and copy its numbers exactly. "
        "Answer in one or two short sentences. Never mention 'the context' or 'the handbook text'. "
        "If the handbook text does not answer the question, reply exactly: "
        "\"I don't know.\""
    )


@lru_cache
def get_settings():
    return Settings()

def get_embeddings(settings: Settings):
    return init_embeddings(
        model=settings.embed_model,
        provider=settings.embed_provider
    )

def get_qdrant_client(settings: Settings):
    return QdrantClient(
        path=settings.qdrant_data
    )