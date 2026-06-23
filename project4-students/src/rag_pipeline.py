from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from .config import ANSWER_MAX_TOKENS, LLM_MODEL, MAX_HISTORY_TURNS, TOP_K, OPENROUTER_BASE_URL
from .safety import validate_query
from .vector_store import query_collection

SYSTEM_PROMPT = """You are a source-grounded RAG assistant for a COMP4451 project.
Answer only using the retrieved context chunks and the brief conversation history.
If the retrieved context does not contain enough evidence, say: "I cannot answer this from the provided documents."
Do not use outside knowledge. Do not invent citations, numbers, lists, or definitions.
For numbers, dates, requirements, and named methods, anchor the answer to the retrieved text.
When the question asks for a list or categories, extract every item present in the context and preserve source grouping.
When the question is comparative, compare only the documents represented in the retrieved context.
End with a short "Sources used" list naming the document title and heading for each relied-on chunk.
"""


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "No prior conversation."
    recent = history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in recent:
        user = turn.get("user", "")[:500]
        assistant = turn.get("assistant", "")[:800]
        lines.append(f"User: {user}\nAssistant: {assistant}")
    return "\n\n".join(lines)


def _format_context(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No retrieved chunks."
    blocks = []
    for c in chunks:
        m = c["metadata"]
        blocks.append(
            f"[Chunk {c['rank']}]\n"
            f"Document: {m.get('document_title')}\n"
            f"Heading: {m.get('heading')}\n"
            f"Source path: {m.get('source_path')}\n"
            f"Distance: {c.get('distance'):.4f}\n"
            f"Text:\n{c['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(query: str, history: list[dict[str, str]] | None = None, top_k: int = TOP_K):
    history = history or []
    ok, reason = validate_query(query)
    if not ok:
        return reason, [], history

    chunks = query_collection(query, top_k=top_k)
    if not chunks:
        answer = "I cannot answer this from the provided documents."
        return answer, chunks, history + [{"user": query, "assistant": answer}]

    client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=OPENROUTER_BASE_URL)
    user_prompt = f"""Conversation history:
{_format_history(history)}

Retrieved context:
{_format_context(chunks)}

User question:
{query}

Write a concise, accurate answer grounded only in the retrieved context."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=ANSWER_MAX_TOKENS,
    )
    answer = response.choices[0].message.content or "I cannot answer this from the provided documents."
    return answer, chunks, history + [{"user": query, "assistant": answer}]
