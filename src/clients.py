import hashlib
import json
import logging
from pathlib import Path

from openai import OpenAI

from src.config import Config

logger = logging.getLogger(__name__)

_cfg = Config()

_llm_client = OpenAI(base_url=_cfg.llm_base_url, api_key=_cfg.llm_api_key)
# TEI does not require a key, but the OpenAI SDK needs a non-empty value
_embed_client = OpenAI(base_url=_cfg.embedding_base_url, api_key="not-needed")

_cache_dir = Path(_cfg.cache_dir) / "llm"


def _cache_file(*parts) -> Path:
    # The key covers everything that changes the answer: changing the prompt or the model
    # automatically invalidates old entries.
    key = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()
    return _cache_dir / f"{key}.json"


def llm(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    reasoning_effort: str = "low",
    use_cache: bool = True,
) -> str:
    """Call the LLM and return the text answer.

    Answers are cached on disk (data/cache/llm). Truncated or empty answers are never cached.
    max_tokens includes reasoning tokens for gpt-oss, so keep it generous.
    """
    cache_file = _cache_file(_cfg.llm_model, system, prompt, temperature, reasoning_effort)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["content"]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = _llm_client.chat.completions.create(
        model=_cfg.llm_model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        extra_body={"reasoning_effort": reasoning_effort},
    )
    choice = resp.choices[0]
    content = choice.message.content or ""

    if choice.finish_reason == "length":
        logger.warning("LLM output was truncated (hit max_tokens=%d)", max_tokens)
    elif use_cache and content:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")

    return content


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a list of texts and return vectors in the same order as the input."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _embed_client.embeddings.create(
            model=_cfg.embedding_model,
            input=batch,
        )
        vectors.extend(d.embedding for d in resp.data)
    return vectors