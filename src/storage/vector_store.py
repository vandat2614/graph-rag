import logging
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

BATCH_SIZE = 256

# One collection per kind of searchable item (same idea as LightRAG)
ENTITIES = "entities"
RELATIONS = "relations"
CHUNKS = "chunks"


@dataclass
class Hit:
    id: str  # original id (not the Qdrant point id)
    score: float
    payload: dict


def _point_id(item_id: str) -> str:
    # Qdrant point ids must be unsigned ints or UUIDs, so derive a stable UUID from our string id
    return str(uuid.uuid5(uuid.NAMESPACE_OID, item_id))


class VectorStore:
    """Qdrant-backed vector store. Knows nothing about entities/relations/chunks,
    it only stores (id, vector, payload) and searches by vector."""

    def __init__(self, url: str, api_key: str, vector_size: int):
        self._client = QdrantClient(url=url, api_key=api_key or None)
        self._vector_size = vector_size

    def setup(self, collections: tuple[str, ...] = (ENTITIES, RELATIONS, CHUNKS)) -> None:
        """Create collections if missing. Safe to call multiple times."""
        for name in collections:
            if not self._client.collection_exists(name):
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=self._vector_size, distance=models.Distance.COSINE
                    ),
                )
                logger.info("Created collection '%s'", name)

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)

    def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        points = [
            models.PointStruct(
                id=_point_id(item_id),
                vector=vector,
                payload={**payload, "id": item_id},
            )
            for item_id, vector, payload in zip(ids, vectors, payloads)
        ]
        for i in range(0, len(points), BATCH_SIZE):
            self._client.upsert(collection_name=collection, points=points[i : i + BATCH_SIZE])
        logger.info("Upserted %d points into '%s'", len(points), collection)

    def search(self, collection: str, vector: list[float], top_k: int = 10) -> list[Hit]:
        result = self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            Hit(id=p.payload["id"], score=p.score, payload=p.payload) for p in result.points
        ]

    def get(self, collection: str, ids: list[str]) -> list[dict]:
        """Fetch payloads by original ids (e.g. chunk text by chunk id)."""
        records = self._client.retrieve(
            collection_name=collection,
            ids=[_point_id(i) for i in ids],
            with_payload=True,
        )
        return [r.payload for r in records]