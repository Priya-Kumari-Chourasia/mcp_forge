"""
RAG building blocks: chunk the docs, index the chunks, search them.

This file is only the "R" - retrieval. The AGENTIC part (deciding what to
search for, judging whether the result is good enough, and rewriting the
query when it is not) lives in rag_nodes.py as graph nodes.

Why retrieval at all? Prose API docs can be far longer than what fits in
one LLM request. Instead of sending "the first N characters" and hoping the
endpoint we care about is in there, we index the WHOLE document and pull
out just the passages relevant to one endpoint at a time.

HOW it searches was decided by measurement (docs/EVALUATION.md, 750 labelled
questions, Hit@4 = a relevant chunk is among the 4 returned):
  - with MiniLM (EMBEDDINGS=minilm, the default with a real LLM): HYBRID - the
    MiniLM ranking and the BM25 keyword ranking combined by reciprocal rank
    fusion. Hit@4 0.913, against 0.825 for MiniLM alone.
  - without it (EMBEDDINGS=none, the default offline and in the tests): BM25
    alone. Hit@4 0.912; the bag-of-words embedding used before scored 0.629.
Keywords help because endpoint paths and parameter names are exact tokens
("conversations.list", "petId") that embeddings blur; embeddings help where the
docs describe an endpoint in words without naming it.
"""

import math
import os
import re
import sys
import uuid
from collections import Counter

# ChromaDB needs SQLite 3.35 or newer. Some Linux hosts (including some
# Streamlit Community Cloud images) ship an older one, so when the bundled
# pysqlite3 package is installed it is used in place of the system sqlite3.
try:
    import pysqlite3  # noqa: F401
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb  # noqa: E402

CHUNK_SIZE = 1200      # characters per chunk
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks


def chunk_spans(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Where each chunk starts and ends in the text: [(start, end), ...].

    Chunks overlap, and each one is cut at a line break when one is near its
    end, so an endpoint's description is less likely to be sliced in half. The
    overlap means text cut at a boundary still appears whole in one of the two
    chunks. (The positions are what evaluation/ needs to know which chunk holds
    which part of the docs.)"""
    spans, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            line_break = text.rfind("\n", start + size // 2, end)
            if line_break != -1:
                end = line_break
        if text[start:end].strip():
            spans.append((start, end))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return spans


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """The chunks themselves (see chunk_spans), with surrounding whitespace removed."""
    return [text[start:end].strip() for start, end in chunk_spans(text, size, overlap)]


# ----------------------------------------------------------------------
# Keyword ranking: BM25
# ----------------------------------------------------------------------

def tokenize(text: str) -> list:
    """Lowercase words; camelCase and path segments are split, so "petId" and
    "/pet/{petId}" both give "pet" and "id"."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Okapi BM25, the standard keyword ranking (k1=1.5, b=0.75). A chunk scores
    higher for each query word it contains, more for rare words (idf), with
    diminishing returns for repeats (k1) and a correction for long chunks (b).
    idf uses the +1 form (as Lucene does), so a word found in most chunks still
    scores a little instead of going negative."""

    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.average = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0
        frequency = Counter(word for d in self.docs for word in d)
        n = len(self.docs)
        self.idf = {w: math.log((n - f + 0.5) / (f + 0.5) + 1) for w, f in frequency.items()}

    def scores(self, query: str) -> list:
        words = tokenize(query)
        result = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            for w in words:
                tf = doc.get(w, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / (self.average or 1))
                    score += self.idf[w] * tf * (self.k1 + 1) / norm
            result.append(score)
        return result


def rank(scores) -> list:
    """Indices by score, best first; ties go to the earlier chunk, so the result
    never depends on luck."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def reciprocal_rank_fusion(rankings, k: int = 60) -> list:
    """Combines rankings: each item scores sum(1 / (k + its rank)). It uses only
    the ranks, so two scores on different scales (cosine, BM25) need no
    calibration. k=60 is the value from the original paper (Cormack et al., 2009)."""
    scores = Counter()
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] += 1 / (k + position)
    n = max((len(r) for r in rankings), default=0)
    return rank([scores[i] for i in range(n)])


# ----------------------------------------------------------------------
# Semantic ranking: MiniLM embeddings
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


def use_minilm() -> bool:
    """EMBEDDINGS=minilm (the default with a real LLM) or EMBEDDINGS=none (keyword
    search only: the default offline, so tests and the stub need no download)."""
    default = "minilm" if os.getenv("GROQ_API_KEY") and os.getenv("USE_REAL_LLM", "true").lower() == "true" else "none"
    return os.getenv("EMBEDDINGS", default).lower() == "minilm"


# ----------------------------------------------------------------------
# The index
# ----------------------------------------------------------------------

class DocIndex:
    """The documentation, chunked, with a BM25 index and (when MiniLM is used)
    the chunks' embeddings in an in-memory ChromaDB collection."""

    def __init__(self, text: str):
        self.chunks = chunk_text(text)
        self.keyword = BM25(self.chunks)
        self.collection = None
        if self.chunks and use_minilm():
            try:
                vectors = _embed_minilm(self.chunks)
            except Exception as e:      # e.g. no internet to download the model
                print(f"[rag] MiniLM embeddings unavailable ({type(e).__name__}); using keyword search (BM25) only")
                os.environ["EMBEDDINGS"] = "none"
                return
            self.collection = chromadb.EphemeralClient().create_collection(
                name=f"docs-{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
            self.collection.add(ids=[str(i) for i in range(len(self.chunks))], documents=self.chunks,
                                embeddings=vectors)

    def search(self, query: str, k: int = 4):
        """Returns the k chunks that best match the query, best first."""
        if not self.chunks:
            return []
        ranking = rank(self.keyword.scores(query))
        if self.collection is not None:
            # The whole MiniLM ranking (not just its top k) goes into the fusion.
            result = self.collection.query(query_embeddings=_embed_minilm([query]), n_results=len(self.chunks))
            ranking = reciprocal_rank_fusion([[int(i) for i in result["ids"][0]], ranking])
        return [self.chunks[i] for i in ranking[:k]]
