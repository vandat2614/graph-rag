from src.clients import llm
from src.config import Config
from src.models import RetrievedContext

_cfg = Config()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context.

Rules:
- Base your answer strictly on the context. Do not use outside knowledge.
- If the context does not contain enough information, say so clearly instead of guessing.
- Be concise and well organized.
- Answer in {language}."""

USER_PROMPT = """Context:

{context}

Question: {question}"""


def format_context(ctx: RetrievedContext) -> str:
    """Turn a RetrievedContext into plain text for the prompt."""
    names = {e.id: e.name for e in ctx.entities}

    entity_lines = [f"- {e.name} ({e.type}): {e.description}" for e in ctx.entities]
    relation_lines = [
        f"- {names.get(r.source_id, r.source_id)} <-> {names.get(r.target_id, r.target_id)} "
        f"[{r.keywords}]: {r.description}"
        for r in ctx.relations
    ]
    chunk_blocks = [f"[{c.doc_id}]\n{c.text}" for c in ctx.chunks]

    return (
        "## Entities\n" + "\n".join(entity_lines)
        + "\n\n## Relations\n" + "\n".join(relation_lines)
        + "\n\n## Source passages\n" + "\n\n".join(chunk_blocks)
    )


def generate_answer(question: str, ctx: RetrievedContext) -> str:
    """Ask the LLM to answer the question from the retrieved context."""
    return llm(
        USER_PROMPT.format(context=format_context(ctx), question=question),
        system=SYSTEM_PROMPT.format(language=_cfg.language),
    )