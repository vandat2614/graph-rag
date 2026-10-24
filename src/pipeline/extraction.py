import hashlib
import json
import logging
import re

from src.clients import llm
from src.config import Config
from src.models import Chunk, Entity, Relation

logger = logging.getLogger(__name__)
_cfg = Config()

SYSTEM_PROMPT = """You are a Knowledge Graph Specialist. Extract entities and relationships from the input text.

Return ONLY a valid JSON object, with no introduction, explanation or markdown.

Schema:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "keywords": "...", "description": "..."}}
  ]
}}

Rules:
1. Entities
   - Extract clearly defined and meaningful entities only.
   - "name": use title case if the name is case-insensitive. Keep naming consistent across entities and relations.
   - "type": one of [{entity_types}]. If none applies, use "Other".
   - "description": concise, based solely on the input text, written in the third person. Do not use pronouns such as "this article", "he", "she", "we".
2. Relations
   - Only between entities that you listed in "entities". "source" and "target" must match an entity name exactly.
   - Only direct, clearly stated relationships. Decompose N-ary relationships into binary pairs.
   - Relations are undirected. Do not output duplicates (A-B and B-A are the same).
   - "keywords": high-level keywords summarizing the nature or theme of the relationship, separated by commas.
   - "description": concise explanation of why the two entities are connected.
   - Put the most significant relations first.
3. Language
   - Write names, keywords and descriptions in {language}.
   - Keep proper nouns (people, places, organizations) in their original language when no widely accepted translation exists.
"""

USER_PROMPT = """Extract entities and relations from the text below.

<text>
{text}
</text>"""

_SYSTEM = SYSTEM_PROMPT.format(
    entity_types=", ".join(_cfg.entity_types),
    language=_cfg.language,
)


def _entity_id(name: str) -> str:
    return "ent-" + hashlib.md5(name.strip().lower().encode()).hexdigest()[:12]


def _relation_id(source_id: str, target_id: str) -> str:
    # Sorted so that A-B and B-A produce the same id (undirected)
    a, b = sorted([source_id, target_id])
    return "rel-" + hashlib.md5(f"{a}|{b}".encode()).hexdigest()[:12]


def _parse_json(raw: str) -> dict | None:
    # Take the outermost {...} to tolerate code fences or extra text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def extract_from_chunk(chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
    """Extract raw entities and relations from one chunk.

    Entities/relations with the same id across chunks are merged in a later stage.
    """
    raw = llm(USER_PROMPT.format(text=chunk.text), system=_SYSTEM)
    data = _parse_json(raw)
    if data is None:
        logger.warning("Chunk %s: invalid JSON from LLM, skipped", chunk.id)
        return [], []

    entities: dict[str, Entity] = {}
    for item in data.get("entities", []):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        entity_id = _entity_id(name)
        entities[entity_id] = Entity(
            id=entity_id,
            name=name,
            type=str(item.get("type") or "Other").strip(),
            description=str(item.get("description", "")).strip(),
            chunk_ids=[chunk.id],
        )

    relations: dict[str, Relation] = {}
    for item in data.get("relations", []):
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if not source or not target:
            continue
        source_id, target_id = _entity_id(source), _entity_id(target)
        if source_id == target_id:
            continue
        if source_id not in entities or target_id not in entities:
            logger.debug("Chunk %s: relation %s -> %s references unknown entity, skipped", chunk.id, source, target)
            continue
        relation_id = _relation_id(source_id, target_id)
        relations[relation_id] = Relation(
            id=relation_id,
            source_id=source_id,
            target_id=target_id,
            keywords=str(item.get("keywords", "")).strip(),
            description=str(item.get("description", "")).strip(),
            chunk_ids=[chunk.id],
        )

    logger.info("Chunk %s: %d entities, %d relations", chunk.id, len(entities), len(relations))
    return list(entities.values()), list(relations.values())