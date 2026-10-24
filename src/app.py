import logging
from pathlib import Path

from src.config import Config
from src.models import RetrievedContext
from src.pipeline.chunking import chunk_document
from src.pipeline.extraction import extract_from_chunk
from src.pipeline.generation import generate_answer
from src.pipeline.indexing import index
from src.pipeline.loader import extract_text_from_pdf
from src.pipeline.merge import merge_entities, merge_extractions, merge_relations
from src.pipeline.retrieval import retrieve
from src.storage.graph_store import GraphStore
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class GraphRAG:
    """Facade over the whole pipeline. This is the only thing a CLI or an API needs to call."""

    def __init__(self, config: Config | None = None):
        cfg = config or Config()
        self.graph = GraphStore(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
        self.vectors = VectorStore(cfg.qdrant_url, cfg.qdrant_api_key, cfg.embedding_dim)
        self.graph.setup()
        self.vectors.setup()

    def ingest_pdf(self, path: str) -> dict[str, int]:
        """PDF -> chunks -> entities/relations -> Neo4j + Qdrant."""
        doc_id = Path(path).stem
        chunks = chunk_document(doc_id, extract_text_from_pdf(path))
        logger.info("Document '%s': %d chunks", doc_id, len(chunks))

        results = []
        for i, chunk in enumerate(chunks, start=1):
            try:
                results.append(extract_from_chunk(chunk))
            except Exception:
                # One bad chunk (rate limit, network...) should not abort the whole document
                logger.exception("Chunk %s: extraction failed, skipped", chunk.id)
            if i % 10 == 0 or i == len(chunks):
                logger.info("Extracted %d/%d chunks", i, len(chunks))

        entities, relations = merge_extractions(results)

        # Merge with what is already stored, so multiple documents accumulate
        # instead of overwriting each other (and re-ingesting the same file is harmless)
        entities = merge_entities(entities + self.graph.get_entities([e.id for e in entities]))
        names = {e.id: e.name for e in entities}
        relations = merge_relations(
            relations + self.graph.get_relations([r.id for r in relations]), names
        )

        index(chunks, entities, relations, self.graph, self.vectors)
        return {"chunks": len(chunks), "entities": len(entities), "relations": len(relations)}

    def ask_with_context(self, question: str) -> tuple[str, RetrievedContext]:
        """Answer a question and also return the context the answer was based on."""
        ctx = retrieve(question, self.graph, self.vectors)
        return generate_answer(question, ctx), ctx

    def ask(self, question: str) -> str:
        return self.ask_with_context(question)[0]

    def close(self) -> None:
        self.graph.close()