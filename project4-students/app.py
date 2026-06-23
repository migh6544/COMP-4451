"""
COMP4451 Project 4/5 — Source-Grounded RAG System

This file defines the Gradio web application for the project.

The app provides a complete Retrieval-Augmented Generation (RAG) interface:
1. Rebuilds a persistent Chroma vector index from the local course documents.
2. Accepts user questions through a Gradio chat interface.
3. Retrieves the most relevant document chunks from Chroma.
4. Sends the retrieved context plus the user question to the RAG pipeline.
5. Displays a source-grounded answer.
6. Displays the retrieved source chunks and metadata.
7. Displays a 3-D Plotly visualization of the retrieved chunk embeddings.
8. Maintains lightweight multi-turn memory using Gradio state.

Important:
- The app expects an OpenRouter API key in a local `.env` file.
- The `.env` file should contain:
    OPENROUTER_API_KEY=your_key_here
- Do not submit `.env` to Canvas or GitHub.
"""

from __future__ import annotations

import os

import gradio as gr
from dotenv import find_dotenv, load_dotenv

from src.build_index import build_or_rebuild_index
from src.rag_pipeline import answer_question
from src.visualization import embedding_figure


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Load environment variables from the nearest `.env` file.
# `find_dotenv(usecwd=True)` starts searching from the current working directory,
# which is important because the app is expected to be launched from the project
# folder: `proj4_michael_ghattas`.
load_dotenv(find_dotenv(usecwd=True))


def has_openrouter_key() -> bool:
    """
    Check whether the OpenRouter API key is available in the environment.

    Returns:
        bool:
            True if `OPENROUTER_API_KEY` exists and is non-empty.
            False otherwise.

    Why this helper exists:
        Both indexing and answering require calls to OpenRouter/OpenAI-compatible
        endpoints. Instead of letting the app fail with a lower-level API error,
        this helper lets the UI display a clear message to the user.
    """
    return bool(os.getenv("OPENROUTER_API_KEY"))


def rebuild_index() -> str:
    """
    Rebuild the persistent Chroma index from the project documents.

    This function is connected to the "Rebuild Chroma Index" button in Gradio.

    The assignment requires the project to:
    - Load the documents.
    - Chunk them in code.
    - Embed them using OpenAI/OpenRouter-compatible embeddings.
    - Store chunk text and metadata in Chroma.

    Returns:
        str:
            A status message displayed in the Gradio UI. On success, this message
            reports how many chunks/documents were indexed. On failure due to a
            missing key, it gives the user a direct setup instruction.
    """
    if not has_openrouter_key():
        return (
            "OPENROUTER_API_KEY was not found. "
            "Add it to a .env file before indexing."
        )

    # reset=True clears and rebuilds the collection so the current documents,
    # chunking logic, and metadata are reflected in Chroma.
    return build_or_rebuild_index(reset=True)


def format_source_chunks(chunks: list[dict]) -> str:
    """
    Convert retrieved RAG chunks into Markdown for display in the UI.

    Each retrieved chunk is expected to contain:
    - rank: retrieval rank after similarity search
    - distance: Chroma distance score
    - text: chunk text
    - metadata: dictionary containing document title, heading, source path, etc.

    Args:
        chunks:
            Retrieved chunks returned by the RAG pipeline.

    Returns:
        str:
            Markdown-formatted source list. If no chunks are retrieved, returns
            a short message stating that no source chunks were retrieved.

    Why this matters:
        The assignment requires the app to show relevant source chunks. This
        function makes retrieval transparent by showing the document, heading,
        path, rank, distance, and preview text for every retrieved chunk.
    """
    source_lines: list[str] = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        # The preview is truncated to keep the UI readable. Newlines are replaced
        # with spaces so Markdown blockquote formatting stays clean.
        preview = chunk.get("text", "")[:900].replace("\n", " ")

        source_lines.append(
            f"### Rank {chunk.get('rank')} | distance={chunk.get('distance', 0):.4f}\n"
            f"**Document:** {metadata.get('document_title')}  \n"
            f"**Heading:** {metadata.get('heading')}  \n"
            f"**Source:** `{metadata.get('source_path')}`\n\n"
            f"> {preview}..."
        )

    return "\n\n".join(source_lines) if source_lines else "No source chunks retrieved."


def append_chat_message(
    chat_history: list[dict],
    role: str,
    content: str,
) -> list[dict]:
    """
    Append a message to the Gradio chat history using message-dictionary format.

    Args:
        chat_history:
            Existing Gradio chatbot history.
        role:
            Message role. Expected values are usually "user" or "assistant".
        content:
            Message text to display.

    Returns:
        list[dict]:
            Updated chat history.

    Why this matters:
        Newer Gradio versions expect chatbot data in a messages-style format:
            {"role": "user", "content": "..."}
            {"role": "assistant", "content": "..."}
        Returning old tuple/list-pair format can cause:
            "Data incompatible with messages format..."
    """
    return chat_history + [{"role": role, "content": content}]


def chat_fn(
    message: str,
    chat_history: list[dict] | None,
    memory_state: list[dict] | None,
    top_k: int | float,
):
    """
    Handle one user question from the Gradio interface.

    This is the core UI callback for the RAG application.

    Flow:
    1. Normalize empty Gradio state values.
    2. Check for the OpenRouter API key.
    3. Send the user question, conversation memory, and top-k value to the
       RAG pipeline.
    4. Receive:
        - final answer
        - retrieved chunks
        - updated memory
    5. Format retrieved chunks as Markdown.
    6. Build a 3-D embedding visualization from the retrieved chunks.
    7. Return all UI updates back to Gradio.

    Args:
        message:
            The latest user question from the textbox.
        chat_history:
            Current visible chat messages from the Gradio chatbot.
        memory_state:
            Lightweight conversation memory stored in Gradio state. This is
            separate from the visible chatbot so the RAG pipeline can use prior
            turns as additional context for follow-up questions.
        top_k:
            Number of chunks to retrieve from Chroma.

    Returns:
        tuple:
            (
                cleared_textbox,
                updated_chat_history,
                updated_memory_state,
                sources_markdown,
                plotly_figure
            )

    Notes:
        The answer itself is generated inside `src.rag_pipeline.answer_question`.
        That pipeline is responsible for source-restricted prompting and refusing
        out-of-scope questions when the retrieved documents do not support an
        answer.
    """
    # Gradio may pass None during the first interaction. Normalize to lists.
    chat_history = chat_history or []
    memory_state = memory_state or []

    # Avoid processing empty or whitespace-only messages.
    if not message or not message.strip():
        return "", chat_history, memory_state, "No question entered.", embedding_figure([])

    message = message.strip()

    # If no API key is available, show a clear error in the chat instead of
    # letting indexing/generation fail deeper in the stack.
    if not has_openrouter_key():
        answer = (
            "OPENROUTER_API_KEY was not found. "
            "Add it to a .env file and restart the app."
        )

        updated_chat = append_chat_message(chat_history, "user", message)
        updated_chat = append_chat_message(updated_chat, "assistant", answer)

        return "", updated_chat, memory_state, "", embedding_figure([])

    # Run the RAG pipeline:
    # - retrieve top-k source chunks
    # - build a grounded prompt
    # - generate an answer with the LLM
    # - update lightweight memory for follow-up questions
    answer, chunks, new_memory = answer_question(
        message,
        history=memory_state,
        top_k=int(top_k),
    )

    # Format source chunks for display below the chat.
    sources_md = format_source_chunks(chunks)

    # Build the 3-D Plotly visualization required by the assignment.
    fig = embedding_figure(chunks)

    # Update visible chat using Gradio's message-dictionary format.
    updated_chat = append_chat_message(chat_history, "user", message)
    updated_chat = append_chat_message(updated_chat, "assistant", answer)

    # Clear the textbox by returning "" as the first output.
    return "", updated_chat, new_memory, sources_md, fig


# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------

with gr.Blocks(title="COMP4451 RAG System") as demo:
    """
    Gradio Blocks app layout.

    Components:
    - Markdown title/instructions
    - Button to rebuild the Chroma index
    - Textbox showing index status
    - Chatbot for user/assistant turns
    - Textbox for new questions
    - Slider for top-k retrieval size
    - Markdown panel for retrieved source chunks
    - Plotly panel for 3-D embedding visualization
    - Gradio State for multi-turn memory
    """

    gr.Markdown(
        "# COMP4451 Project 4/5 — Source-Grounded RAG System\n"
        "Ask questions about the indexed Agentic RAG course documents. "
        "The answer is restricted to retrieved chunks and shows sources plus "
        "a 3-D embedding view."
    )

    # Stores conversation memory used by the RAG pipeline. This is not displayed
    # directly, but it lets the user ask follow-up questions such as:
    # "How is that different from ReAct?"
    memory_state = gr.State([])

    # Index controls.
    with gr.Row():
        rebuild_btn = gr.Button("Rebuild Chroma Index")
        rebuild_status = gr.Textbox(label="Index status", interactive=False)

    # Chat display.
    #
    # This Gradio installation expects message dictionaries with "role" and
    # "content" keys. The callback returns that format.
    chatbot = gr.Chatbot(label="RAG Chat", height=420)

    # User input controls.
    with gr.Row():
        msg = gr.Textbox(
            label="Question",
            placeholder="Example: Why does ReAct improve RAG retrieval?",
            scale=5,
        )
        top_k = gr.Slider(
            minimum=3,
            maximum=10,
            value=6,
            step=1,
            label="Top-k retrieved chunks",
            scale=1,
        )

    send = gr.Button("Ask")

    # Output panels.
    sources = gr.Markdown(label="Relevant source chunks")
    plot = gr.Plot(label="3-D embedding visualization")

    # Button event: rebuild vector index.
    rebuild_btn.click(
        rebuild_index,
        outputs=rebuild_status,
    )

    # Button event: ask question.
    send.click(
        chat_fn,
        inputs=[msg, chatbot, memory_state, top_k],
        outputs=[msg, chatbot, memory_state, sources, plot],
    )

    # Enter-key event: ask question.
    msg.submit(
        chat_fn,
        inputs=[msg, chatbot, memory_state, top_k],
        outputs=[msg, chatbot, memory_state, sources, plot],
    )


if __name__ == "__main__":
    """
    Launch the local Gradio app.

    By default, Gradio serves the app locally at:
        http://127.0.0.1:7860

    For this assignment, local launch is sufficient. Public sharing is not needed.
    """
    demo.launch()
