"""
RAG building blocks: chunk the docs, embed the chunks, search them.

This file is only the "R" - retrieval. The AGENTIC part (deciding what to
search for, judging whether the result is good enough, and rewriting the
query when it is not) lives in rag_nodes.py as graph nodes.

Why retrieval at all? Prose API docs can be far longer than what fits in
one LLM request. Instead of sending "the first N characters" and hoping the
endpoint we care about is in there, we index the WHOLE document and pull
out just the passages relevant to one endpoint at a time.
"""

import hashlib
import math
import os
import re
import uuid

import chromadb

CHUNK_SIZE = 1200      # characters per chunk
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Splits text into overlapping chunks, cutting at a line break when one
    is near the end of the chunk, so an endpoint's description is less likely
    to be sliced in half. The overlap means text cut at a boundary still
    appears whole in one of the two chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            line_break = text.rfind("\n", start + size // 2, end)
            if line_break != -1:
                end = line_break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# ----------------------------------------------------------------------
# Embeddings - two interchangeable backends
# ----------------------------------------------------------------------

_minilm = None


def _embed_minilm(texts):
    """all-MiniLM-L6-v2 sentence embeddings (the HuggingFace model, run locally
    through ONNX by ChromaDB). Downloads the model once on first use."""
    global _minilm
    if _minilm is None:
        from chromadb.utils import embedding_functions
        _minilm = embedding_functions.DefaultEmbeddingFunction()
    return [[float(x) for x in vector] for vector in _minilm(list(texts))]


def _embed_hash(texts, dims: int = 512):
    """Offline fallback: a bag-of-words vector. Each word is hashed to one of
    `dims` slots and counted. It only matches on shared words (no notion of
    meaning), but it needs no download, so tests and the offline stub can run
    anywhere."""
    vectors = []
    for text in texts:
        vector = [0.0] * dims
        for word in re.findall(r"[a-z0-9_]+", text.lower()):
            slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % dims
            vector[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


def embed(texts):
    """EMBEDDINGS=minilm (default with a real LLM) or EMBEDDINGS=hash (offline)."""
    default = "minilm" if os.getenv("GROQ_API_KEY") and os.getenv("USE_REAL_LLM", "true").lower() == "true" else "hash"
    backend = os.getenv("EMBEDDINGS", default).lower()
    if backend == "minilm":
        try:
            return _embed_minilm(texts)
        except Exception as e:      # e.g. no internet to download the model
            print(f"[rag] MiniLM embeddings unavailable ({type(e).__name__}); using hash embeddings instead")
            os.environ["EMBEDDINGS"] = "hash"
    return _embed_hash(texts)


# ----------------------------------------------------------------------
# The vector index
# ----------------------------------------------------------------------

class DocIndex:
    """The documentation, chunked and stored in an in-memory ChromaDB collection."""

    def __init__(self, text: str):
        self.chunks = chunk_text(text)
        client = chromadb.EphemeralClient()
        self.collection = client.create_collection(
            name=f"docs-{uuid.uuid4().hex}",
            metadata={"hnsw:space": "cosine"},
        )
        if self.chunks:
            self.collection.add(
                ids=[str(i) for i in range(len(self.chunks))],
                documents=self.chunks,
                embeddings=embed(self.chunks),
            )

    def search(self, query: str, k: int = 4):
        """Returns the k chunks most similar to the query, best first."""
        if not self.chunks:
            return []
        result = self.collection.query(
            query_embeddings=embed([query]),
            n_results=min(k, len(self.chunks)),
        )
        return result["documents"][0]
