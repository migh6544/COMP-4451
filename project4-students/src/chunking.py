from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .document_loader import LoadedDocument
from .config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict


def count_tokens(text: str) -> int:
    """Approximate token count without requiring a tokenizer at runtime."""
    return max(1, int(len(re.findall(r"\w+|[^\w\s]", text)) * 0.75))


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown/text into sections while preserving heading metadata."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Untitled Section"
    current_lines: list[str] = []

    heading_re = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Za-z0-9 ,:&/()\-]{3,}:)$")
    for line in lines:
        stripped = line.strip()
        if stripped and heading_re.match(stripped) and current_lines:
            sections.append((current_heading, current_lines))
            current_heading = stripped.lstrip("#").strip().rstrip(":")
            current_lines = [line]
        else:
            if stripped and heading_re.match(stripped) and current_heading == "Untitled Section":
                current_heading = stripped.lstrip("#").strip().rstrip(":")
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections if "\n".join(lines).strip()]


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def _window_sentences(sentences: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)
        if current and current_tokens + sent_tokens > max_tokens:
            chunks.append(" ".join(current).strip())
            overlap: list[str] = []
            overlap_count = 0
            for prev in reversed(current):
                if overlap_count >= overlap_tokens:
                    break
                overlap.insert(0, prev)
                overlap_count += count_tokens(prev)
            current = overlap[:]
            current_tokens = sum(count_tokens(s) for s in current)
        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def chunk_documents(documents: list[LoadedDocument]) -> list[TextChunk]:
    """Structure-aware chunking: split on headings first, then sentence windows with overlap."""
    all_chunks: list[TextChunk] = []

    for doc in documents:
        sections = _split_by_headings(doc.text)
        chunk_idx = 0
        for heading, section_text in sections:
            if count_tokens(section_text) <= CHUNK_MAX_TOKENS:
                pieces = [section_text]
            else:
                pieces = _window_sentences(
                    _sentence_split(section_text),
                    max_tokens=CHUNK_TARGET_TOKENS,
                    overlap_tokens=CHUNK_OVERLAP_TOKENS,
                )
            for local_idx, piece in enumerate(pieces):
                chunk_id = f"{doc.doc_id}::chunk_{chunk_idx:04d}"
                all_chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        text=piece,
                        metadata={
                            "doc_id": doc.doc_id,
                            "document_title": doc.title,
                            "source_path": doc.source_path,
                            "heading": heading,
                            "chunk_index": chunk_idx,
                            "local_section_chunk": local_idx,
                            "token_count_est": count_tokens(piece),
                        },
                    )
                )
                chunk_idx += 1

    return all_chunks
