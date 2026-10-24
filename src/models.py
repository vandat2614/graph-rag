from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str


@dataclass
class Entity:
    id: str
    name: str
    type: str
    description: str
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class Relation:
    id: str
    source_id: str
    target_id: str
    keywords: str  # chuỗi keyword ngăn cách bằng dấu phẩy, như LightRAG
    description: str
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievedContext:
    entities: list[Entity]
    relations: list[Relation]
    chunks: list[Chunk]