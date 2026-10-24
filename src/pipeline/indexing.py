import logging

from src.clients import embed
from src.models import Chunk, Entity, Relation
from src.storage.graph_store import GraphStore
from src.storage.vector_store import CHUNKS, ENTITIES, RELATIONS, VectorStore

logger = logging.getLogger(__name__)


def _entity_text(e: Entity) -> str:
    # Text that gets embedded for an entity
    return f"{e.name}\n{e.description}"


def _relation_text(r: Relation, names: dict[str, str]) -> str:
    # Text that gets embedded for a relation
    source = names.get(r.source_id, r.source_id)
    target = names.get(r.target_id, r.target_id)
    return f"{r.keywords}\n{source} - {target}\n{r.description}"


def index(
    chunks: list[Chunk],
    entities: list[Entity],
    relations: list[Relation],
    graph: GraphStore,
    vectors: VectorStore,
) -> None:
    """Write extracted data into Neo4j (structure) and Qdrant (embeddings)."""
    names = {e.id: e.name for e in entities}

    # 1. Graph structure
    graph.upsert_entities(entities)
    graph.upsert_relations(relations)

    # 2. Vector indexes
    vectors.upsert(
        CHUNKS,
        ids=[c.id for c in chunks],
        vectors=embed([c.text for c in chunks]),
        payloads=[{"doc_id": c.doc_id, "text": c.text} for c in chunks],
    )
    vectors.upsert(
        ENTITIES,
        ids=[e.id for e in entities],
        vectors=embed([_entity_text(e) for e in entities]),
        payloads=[{"name": e.name, "type": e.type} for e in entities],
    )
    vectors.upsert(
        RELATIONS,
        ids=[r.id for r in relations],
        vectors=embed([_relation_text(r, names) for r in relations]),
        payloads=[
            {"source_id": r.source_id, "target_id": r.target_id, "keywords": r.keywords}
            for r in relations
        ],
    )
    logger.info(
        "Indexed %d chunks, %d entities, %d relations",
        len(chunks), len(entities), len(relations),
    )