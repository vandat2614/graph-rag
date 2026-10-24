from src.config import Config
from src.models import Chunk

_cfg = Config()


def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    """Cắt text của một tài liệu thành các Chunk (cửa sổ trượt theo ký tự)."""
    size, overlap = _cfg.chunk_size, _cfg.chunk_overlap
    step = size - overlap

    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(text), step)):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(Chunk(id=f"{doc_id}-{i}", doc_id=doc_id, text=piece))
        if start + size >= len(text):
            break
    return chunks