import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # LLM (Groq, tương thích OpenAI SDK)
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_api_key: str = os.getenv("GROQ_API_KEY", "")
    cache_dir: str = "./data/cache"
    
    # Embedding (TEI trên VM, tương thích OpenAI)
    embedding_dim: int = 1024
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "http://168.107.88.236:8081/v1")

    chunk_size: int = 1200
    chunk_overlap: int = 100
    data_dir: str = "./data"

    # Extraction
    entity_types: tuple[str, ...] = ("Person", "Organization", "Location", "Event", "Concept", "Other")
    language: str = "Vietnamese"

    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")