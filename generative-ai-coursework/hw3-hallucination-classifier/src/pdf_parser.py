"""Open-source PDF parsing and chunk caching."""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

from .config import CHUNK_MAX_WORDS, CHUNK_MIN_WORDS

fitz.TOOLS.mupdf_display_errors(False)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _split_long_text(text: str, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _page_blocks(page: fitz.Page) -> list[str]:
    blocks = page.get_text("blocks")
    sorted_blocks = sorted(blocks, key=lambda block: (round(block[1], 1), round(block[0], 1)))
    texts = []
    for block in sorted_blocks:
        if len(block) < 5:
            continue
        text = normalize_text(str(block[4]))
        if _word_count(text) >= 5:
            texts.append(text)
    return texts


def parse_pdf_to_chunks(
    pdf_path: str | Path,
    *,
    min_words: int = CHUNK_MIN_WORDS,
    max_words: int = CHUNK_MAX_WORDS,
) -> list[dict]:
    pdf_path = Path(pdf_path)
    chunks: list[dict] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            buffer: list[str] = []
            buffer_words = 0
            for block_text in _page_blocks(page):
                block_parts = _split_long_text(block_text, max_words)
                for part in block_parts:
                    count = _word_count(part)
                    if buffer and buffer_words + count > max_words:
                        chunk_text = normalize_text(" ".join(buffer))
                        chunks.append(
                            {
                                "paper_id": pdf_path.stem,
                                "page": page_index,
                                "chunk_id": len(chunks),
                                "text": chunk_text,
                                "word_count": _word_count(chunk_text),
                            }
                        )
                        buffer = []
                        buffer_words = 0
                    buffer.append(part)
                    buffer_words += count
                    if buffer_words >= min_words:
                        chunk_text = normalize_text(" ".join(buffer))
                        chunks.append(
                            {
                                "paper_id": pdf_path.stem,
                                "page": page_index,
                                "chunk_id": len(chunks),
                                "text": chunk_text,
                                "word_count": _word_count(chunk_text),
                            }
                        )
                        buffer = []
                        buffer_words = 0
            if buffer:
                chunk_text = normalize_text(" ".join(buffer))
                chunks.append(
                    {
                        "paper_id": pdf_path.stem,
                        "page": page_index,
                        "chunk_id": len(chunks),
                        "text": chunk_text,
                        "word_count": _word_count(chunk_text),
                    }
                )
    return [chunk for chunk in chunks if chunk["word_count"] >= 5]


def chunk_cache_path(cache_dir: str | Path, split: str, paper_id: str) -> Path:
    return Path(cache_dir) / "chunks" / split / f"{paper_id}.jsonl"


def load_chunks(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_chunks(path: str | Path, chunks: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def ensure_chunks(pdf_path: str | Path, cache_dir: str | Path, split: str, force: bool = False) -> list[dict]:
    pdf_path = Path(pdf_path)
    cache_path = chunk_cache_path(cache_dir, split, pdf_path.stem)
    if cache_path.exists() and not force:
        return load_chunks(cache_path)
    chunks = parse_pdf_to_chunks(pdf_path)
    if not chunks:
        raise ValueError(f"No text chunks extracted from {pdf_path}")
    save_chunks(cache_path, chunks)
    return chunks
