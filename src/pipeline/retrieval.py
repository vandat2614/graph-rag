import logging
from collections import Counter

from src.clients import embed
from src.models import Chunk, RetrievedContext
from src.storage.graph_store import GraphStore
from src.storage.vector_store import CHUNKS, ENTITIES, RELATIONS, VectorStore

logger = logging.getLogger(__name__)


def _in_order(items: list, ids: list[str]) -> list:
    """Return items sorted by the order of `ids`, dropping ids that were not found."""
    by_id = {item.id: item for item in items}
    return [by_id[i] for i in ids if i in by_id]


def retrieve(
    question: str,
    graph: GraphStore,
    vectors: VectorStore,
    top_k: int = 10,
    max_relations: int = 30,
    max_chunks: int = 6,
) -> RetrievedContext:
    """Hybrid retrieval: vector search finds starting points, the graph expands them."""
    query_vector = embed([question])[0]

    # 1. Vector search: entry points (by meaning)
    entity_hits = vectors.search(ENTITIES, query_vector, top_k)
    relation_hits = vectors.search(RELATIONS, query_vector, top_k)
    chunk_hits = vectors.search(CHUNKS, query_vector, (max_chunks + 1) // 2)

    # 2. Relations: direct hits first, then relations touching the hit entities
    relation_ids = [h.id for h in relation_hits]
    direct = _in_order(graph.get_relations(relation_ids), relation_ids)
    expanded = graph.get_relations_of_entities([h.id for h in entity_hits])
    relations = list({r.id: r for r in direct + expanded}.values())[:max_relations]

    # 3. Entities: hit entities plus the endpoints of the selected relations
    entity_ids = list(
        dict.fromkeys(
            [h.id for h in entity_hits]
            + [i for r in relations for i in (r.source_id, r.target_id)]
        )
    )
    entities = _in_order(graph.get_entities(entity_ids), entity_ids)

    # 4. Chunks: vector hits first, then chunks most referenced by the selected entities/relations
    counts = Counter(cid for x in [*entities, *relations] for cid in x.chunk_ids)
    vector_chunk_ids = [h.id for h in chunk_hits]
    linked_chunk_ids = [cid for cid, _ in counts.most_common() if cid not in vector_chunk_ids]
    chunk_ids = (vector_chunk_ids + linked_chunk_ids)[:max_chunks]

    payloads = {p["id"]: p for p in vectors.get(CHUNKS, chunk_ids)}
    chunks = [
        Chunk(id=cid, doc_id=payloads[cid]["doc_id"], text=payloads[cid]["text"])
        for cid in chunk_ids
        if cid in payloads
    ]

    logger.info(
        "Retrieved %d entities, %d relations, %d chunks",
        len(entities), len(relations), len(chunks),
    )
    return RetrievedContext(entities=entities, relations=relations, chunks=chunks)