import logging
from collections import Counter, defaultdict

from src.clients import llm
from src.config import Config
from src.models import Entity, Relation

logger = logging.getLogger(__name__)
_cfg = Config()

# If the combined description is longer than this, summarize it with the LLM
MAX_DESCRIPTION_CHARS = 1000

SUMMARIZE_PROMPT = """Merge the following descriptions of "{label}" into ONE concise, coherent description.

Rules:
- Use only the information given. Do not add anything new.
- Write in the third person, without pronouns such as "this", "he", "she".
- Write in {language}.
- Output only the merged description, with no introduction.

Descriptions:
{descriptions}"""


def _merge_descriptions(label: str, descriptions: list[str]) -> str:
    # Keep unique non-empty descriptions in their original order
    unique = list(dict.fromkeys(d.strip() for d in descriptions if d.strip()))
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]

    combined = " ".join(unique)
    if len(combined) <= MAX_DESCRIPTION_CHARS:
        return combined

    logger.info("Summarizing %d descriptions for '%s'", len(unique), label)
    prompt = SUMMARIZE_PROMPT.format(
        label=label,
        language=_cfg.language,
        descriptions="\n".join(f"- {d}" for d in unique),
    )
    summary = llm(prompt).strip()
    # Fall back to plain concatenation if the LLM returns nothing
    return summary or combined


def _union(lists: list[list[str]]) -> list[str]:
    # Union of lists, keeping first-seen order
    return list(dict.fromkeys(x for items in lists for x in items))


def merge_entities(entities: list[Entity]) -> list[Entity]:
    groups: dict[str, list[Entity]] = defaultdict(list)
    for e in entities:
        groups[e.id].append(e)

    merged: list[Entity] = []
    for entity_id, items in groups.items():
        name = items[0].name
        merged.append(
            Entity(
                id=entity_id,
                name=name,
                type=Counter(e.type for e in items).most_common(1)[0][0],
                description=_merge_descriptions(name, [e.description for e in items]),
                chunk_ids=_union([e.chunk_ids for e in items]),
            )
        )
    return merged


def merge_relations(relations: list[Relation], names: dict[str, str]) -> list[Relation]:
    """Merge relations with the same id. `names` maps entity id -> entity name (for logging/prompts)."""
    groups: dict[str, list[Relation]] = defaultdict(list)
    for r in relations:
        groups[r.id].append(r)

    merged: list[Relation] = []
    for relation_id, items in groups.items():
        first = items[0]
        label = f"{names.get(first.source_id, first.source_id)} - {names.get(first.target_id, first.target_id)}"
        keywords = list(
            dict.fromkeys(
                k.strip()
                for r in items
                for k in r.keywords.split(",")
                if k.strip()
            )
        )
        merged.append(
            Relation(
                id=relation_id,
                source_id=first.source_id,
                target_id=first.target_id,
                keywords=", ".join(keywords),
                description=_merge_descriptions(label, [r.description for r in items]),
                chunk_ids=_union([r.chunk_ids for r in items]),
            )
        )
    return merged


def merge_extractions(
    results: list[tuple[list[Entity], list[Relation]]],
) -> tuple[list[Entity], list[Relation]]:
    """Merge per-chunk extraction results into one deduplicated set."""
    all_entities = [e for entities, _ in results for e in entities]
    all_relations = [r for _, relations in results for r in relations]

    entities = merge_entities(all_entities)
    names = {e.id: e.name for e in entities}
    relations = merge_relations(all_relations, names)

    logger.info(
        "Merged %d -> %d entities, %d -> %d relations",
        len(all_entities), len(entities), len(all_relations), len(relations),
    )
    return entities, relations