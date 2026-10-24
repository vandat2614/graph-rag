import logging

from neo4j import GraphDatabase

from src.models import Entity, Relation

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


class GraphStore:
    """Neo4j-backed graph store.

    Entities are (:Entity) nodes, relations are [:RELATED] edges.
    Relations are undirected in our model; Neo4j stores one direction
    (smaller id -> larger id) and we query without caring about direction.
    """

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def setup(self) -> None:
        """Create constraints/indexes. Safe to call multiple times."""
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )

    # ---------- helpers ----------

    def _write_batches(self, query: str, rows: list[dict]) -> None:
        with self._driver.session() as session:
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i : i + BATCH_SIZE]
                session.execute_write(lambda tx, b=batch: tx.run(query, rows=b).consume())

    def _read(self, query: str, **params) -> list[dict]:
        with self._driver.session() as session:
            return session.execute_read(lambda tx: tx.run(query, **params).data())

    # ---------- write ----------

    def upsert_entities(self, entities: list[Entity]) -> None:
        rows = [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type,
                "description": e.description,
                "chunk_ids": e.chunk_ids,
            }
            for e in entities
        ]
        self._write_batches(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {id: row.id})
            SET e.name = row.name,
                e.type = row.type,
                e.description = row.description,
                e.chunk_ids = row.chunk_ids
            """,
            rows,
        )
        logger.info("Upserted %d entities", len(rows))

    def upsert_relations(self, relations: list[Relation]) -> None:
        rows = []
        for r in relations:
            # Normalize direction so the same undirected relation never becomes two edges
            source_id, target_id = sorted([r.source_id, r.target_id])
            rows.append(
                {
                    "id": r.id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "keywords": r.keywords,
                    "description": r.description,
                    "chunk_ids": r.chunk_ids,
                }
            )
        self._write_batches(
            """
            UNWIND $rows AS row
            MATCH (s:Entity {id: row.source_id}), (t:Entity {id: row.target_id})
            MERGE (s)-[r:RELATED {id: row.id}]->(t)
            SET r.keywords = row.keywords,
                r.description = row.description,
                r.chunk_ids = row.chunk_ids
            """,
            rows,
        )
        logger.info("Upserted %d relations", len(rows))

    # ---------- read ----------

    def get_entities(self, ids: list[str]) -> list[Entity]:
        records = self._read(
            """
            MATCH (e:Entity) WHERE e.id IN $ids
            RETURN e.id AS id, e.name AS name, e.type AS type,
                   e.description AS description, e.chunk_ids AS chunk_ids
            """,
            ids=ids,
        )
        return [Entity(**r) for r in records]

    def get_relations(self, ids: list[str]) -> list[Relation]:
        records = self._read(
            """
            MATCH (s:Entity)-[r:RELATED]->(t:Entity) WHERE r.id IN $ids
            RETURN r.id AS id, s.id AS source_id, t.id AS target_id,
                   r.keywords AS keywords, r.description AS description,
                   r.chunk_ids AS chunk_ids
            """,
            ids=ids,
        )
        return [Relation(**r) for r in records]

    def get_relations_of_entities(self, entity_ids: list[str]) -> list[Relation]:
        """All relations that touch at least one of the given entities (either direction)."""
        records = self._read(
            """
            MATCH (s:Entity)-[r:RELATED]->(t:Entity)
            WHERE s.id IN $ids OR t.id IN $ids
            RETURN r.id AS id, s.id AS source_id, t.id AS target_id,
                   r.keywords AS keywords, r.description AS description,
                   r.chunk_ids AS chunk_ids
            """,
            ids=entity_ids,
        )
        return [Relation(**r) for r in records]