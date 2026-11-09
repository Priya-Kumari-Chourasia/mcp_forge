"""
RETRIEVAL EVALUATION - how often does retrieval find the chunk that
describes an endpoint?

    python -m evaluation.retrieval_eval                 # all retrievers available here
    python -m evaluation.retrieval_eval --only bm25,hash
    python -m evaluation.retrieval_eval --only hash+bm25

For every endpoint in every document (evaluation/datasets.py) it asks the
pipeline's own first-round question - rag_nodes.first_round_query(), e.g.
"GET /pet/{petId}" - and an alternative (below), over the pipeline's own
chunks (rag.chunk_spans), and checks where the relevant chunk lands in each
retriever's ranking. This is retrieval only: no LLM, no grading,
no query rewriting. It measures how good the FIRST round is, i.e. how much work
is left for the agentic part.

Two first-round queries are compared (an ablation):
  path-only      "METHOD /path": what the pipeline sends (rag_nodes.first_round_query)
  generic-words  "METHOD /path endpoint description parameters base URL": what it
                 sent before this evaluation showed the extra words hurt

Retrievers compared:
  random   the expected score of a random ranking: the floor to beat
  hash     bag-of-words vectors: the offline fallback the pipeline used before
  bm25     rag.BM25: what the pipeline uses without MiniLM
  minilm   all-MiniLM-L6-v2 sentence embeddings alone (downloads the model once;
           skipped if unavailable)
  hybrid   minilm and bm25 combined by rag.reciprocal_rank_fusion: what the
           pipeline uses with MiniLM
  hash+bm25  hash and bm25 combined the same way. Not in the default run: it
           scored below bm25 alone, so the pipeline does not use it. Ask for it
           with --only hash+bm25 to reproduce that number.
The ranking code is imported from rag.py, so what is measured is what runs.

Metrics, averaged over questions (k = 4, the pipeline's TOP_K):
  Hit@1     the best-ranked chunk is relevant
  Hit@4     at least one relevant chunk is in the top 4 - what the grader needs
  Recall@4  the share of the relevant chunks that are in the top 4 (a key
            sentence can lie in two overlapping chunks)
  MRR       mean of 1 / (rank of the first relevant chunk)

Results go to evaluation/results/retrieval.json, with every miss listed for
error analysis. A partial run (--only) writes its own file instead, such as
evaluation/results/retrieval_bm25_hash.json, so it never overwrites the full results.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import time
from collections import Counter

from evaluation import datasets
from rag import (CHUNK_OVERLAP, CHUNK_SIZE, BM25, _embed_minilm, chunk_spans, rank as _rank,
                 reciprocal_rank_fusion, tokenize)
from rag_nodes import TOP_K, first_round_query

RESULTS_PATH = os.path.join(datasets.HERE, "results", "retrieval.json")
QUERIES = {"path-only": first_round_query,
           "generic-words": lambda method, path: f"{method} {path} endpoint description parameters base URL"}
GROUPS = ("hand-written", "reference", "narrative")


# ----------------------------------------------------------------------
# Retrievers. Each returns the chunk indices, best first.
# ----------------------------------------------------------------------

def hash_embed(texts, dims: int = 512):
    """The bag-of-words embedding the pipeline used offline before this evaluation:
    each word is hashed to one of `dims` slots and counted. Kept as a baseline."""
    vectors = []
    for text in texts:
        vector = [0.0] * dims
        for word in re.findall(r"[a-z0-9_]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
        vectors.append(vector)
    return vectors


_query_vectors = {}       # (embedding function, query) -> unit vector, shared by every document and ranker


def _query_vector(embed, query):
    """Each query is embedded once. With MiniLM an embedding is a neural network
    run, and the same queries are asked by minilm and hybrid, so this matters:
    embedding the query again for every chunk took ~50 times as many model calls."""
    key = (embed, query)
    if key not in _query_vectors:
        _query_vectors[key] = _unit(embed([query])[0])
    return _query_vectors[key]


def _cosine_ranker(embed):
    def ranker(chunks):
        vectors = [_unit(v) for v in embed(chunks)]          # all of a document's chunks in one call

        def rank(query):
            q = _query_vector(embed, query)
            return _rank([sum(a * b for a, b in zip(q, v)) for v in vectors])
        return rank
    return ranker


def _unit(vector):
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def _bm25_ranker(chunks):
    index = BM25(chunks)
    return lambda query: _rank(index.scores(query))


def _hybrid_ranker(chunks):
    dense, keyword = _cosine_ranker(_embed_minilm)(chunks), _bm25_ranker(chunks)
    return lambda query: reciprocal_rank_fusion([dense(query), keyword(query)])


def _hash_bm25_ranker(chunks):
    """The hash embedding and BM25 combined by reciprocal rank fusion. Not used by
    the pipeline: it scored below BM25 alone (docs/EVALUATION.md), which is why the
    offline mode uses BM25 alone. Kept so that number can be reproduced."""
    dense, keyword = _cosine_ranker(hash_embed)(chunks), _bm25_ranker(chunks)
    return lambda query: reciprocal_rank_fusion([dense(query), keyword(query)])


RANKERS = {"hash": _cosine_ranker(hash_embed), "bm25": _bm25_ranker,
           "minilm": _cosine_ranker(_embed_minilm), "hybrid": _hybrid_ranker,
           "hash+bm25": _hash_bm25_ranker}

# What a plain `python -m evaluation.retrieval_eval` runs. hash+bm25 is left out of
# the main table; ask for it with --only hash+bm25.
DEFAULT_RETRIEVERS = "random,hash,bm25,minilm,hybrid"


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def relevant_chunks(spans, key) -> set:
    """The chunks that contain the key sentence whole."""
    start, end = key
    return {i for i, (s, e) in enumerate(spans) if s <= start and end <= e}


def score_ranking(ranking, relevant, k: int = TOP_K) -> dict:
    first = next(position for position, i in enumerate(ranking, start=1) if i in relevant)
    top = set(ranking[:k])
    return {"hit@1": float(ranking[0] in relevant), f"hit@{k}": float(bool(top & relevant)),
            f"recall@{k}": len(top & relevant) / len(relevant), "mrr": 1 / first, "first_rank": first}


def expected_random(n_chunks: int, n_relevant: int, k: int = TOP_K) -> dict:
    """Exact expected scores of a uniformly random ranking (no simulation)."""
    n, r = n_chunks, n_relevant
    miss = lambda depth: math.comb(n - r, depth) / math.comb(n, depth)         # no relevant chunk in the top `depth`
    p_first = lambda position: miss(position - 1) * r / (n - position + 1)      # first relevant chunk at `position`
    return {"hit@1": r / n, f"hit@{k}": 1 - miss(min(k, n)), f"recall@{k}": min(k, n) / n,
            "mrr": sum(p_first(p) / p for p in range(1, n - r + 2)), "first_rank": None}


def _mean(rows, metric):
    return sum(row[metric] for row in rows) / len(rows) if rows else float("nan")


# ----------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------

def available(name: str) -> bool:
    """MiniLM needs its model, downloaded on first use. Without it, skip it (and hybrid)."""
    if name not in ("minilm", "hybrid"):
        return True
    try:
        _embed_minilm(["test"])
        return True
    except Exception as e:
        print(f"[skip] {name}: the MiniLM model is not available here ({type(e).__name__})")
        return False


def evaluate(documents, retrievers, queries=("path-only",), k: int = TOP_K) -> dict:
    """Scores every (retriever, query) pair on every question. Rows are named
    "bm25 / path-only" and so on; random does not depend on the query."""
    runs = ["random"] if "random" in retrievers else []
    runs += [f"{name} / {query}" for name in retrievers if name != "random" for query in queries]
    per_question, misses = {run: [] for run in runs}, {run: [] for run in runs}
    for number, doc in enumerate(documents, start=1):
        started = time.time()
        spans = chunk_spans(doc["text"])
        chunks = [doc["text"][s:e].strip() for s, e in spans]
        rankers = {name: RANKERS[name](chunks) for name in retrievers if name != "random"}
        for q in doc["questions"]:
            relevant = relevant_chunks(spans, q["key"])
            for run in runs:
                if run == "random":
                    row = expected_random(len(chunks), len(relevant), k)
                else:
                    name, query_name = run.split(" / ")
                    row = score_ranking(rankers[name](QUERIES[query_name](q["method"], q["path"])), relevant, k)
                    if not row[f"hit@{k}"]:
                        misses[run].append({"document": doc["name"], "endpoint": f"{q['method']} {q['path']}",
                                            "first_relevant_rank": row["first_rank"], "chunks": len(chunks)})
                per_question[run].append({**row, "group": doc["style"], "document": doc["name"]})
        print(f"[{number}/{len(documents)}] {doc['name']}: {len(doc['questions'])} questions, {len(chunks)} chunks "
              f"({time.time() - started:.1f} s)", flush=True)

    metrics = ("hit@1", f"hit@{k}", f"recall@{k}", "mrr")
    summary = {}
    for name, rows in per_question.items():
        summary[name] = {"all": {m: _mean(rows, m) for m in metrics} | {"questions": len(rows)}}
        for group in GROUPS:
            subset = [r for r in rows if r["group"] == group]
            summary[name][group] = {m: _mean(subset, m) for m in metrics} | {"questions": len(subset)}
        for document in dict.fromkeys(r["document"] for r in rows):
            subset = [r for r in rows if r["document"] == document]
            summary[name].setdefault("documents", {})[document] = {m: _mean(subset, m) for m in metrics}
    return {"summary": summary, "misses": misses}


def markdown_table(summary: dict, k: int = TOP_K) -> str:
    lines = [f"| Retriever / query | Questions | Hit@1 | Hit@{k} | Recall@{k} | MRR | Hit@{k} hand-written | "
             f"Hit@{k} reference | Hit@{k} narrative |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        a = s["all"]
        lines.append(f"| {name} | {a['questions']} | {a['hit@1']:.3f} | {a[f'hit@{k}']:.3f} | {a[f'recall@{k}']:.3f} | "
                     f"{a['mrr']:.3f} | {s['hand-written'][f'hit@{k}']:.3f} | {s['reference'][f'hit@{k}']:.3f} | "
                     f"{s['narrative'][f'hit@{k}']:.3f} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", help=f"comma-separated retrievers (default: {DEFAULT_RETRIEVERS}; "
                                       f"also available: hash+bm25)")
    args = parser.parse_args(argv)
    wanted = (args.only or DEFAULT_RETRIEVERS).split(",")
    unknown = [name for name in wanted if name != "random" and name not in RANKERS]
    if unknown:
        parser.error(f"unknown retriever(s) {unknown}; choose from random, {', '.join(RANKERS)}")
    retrievers = [name for name in wanted if available(name)]
    # A partial run (--only) gets its own file, so it never overwrites the full
    # results - the MiniLM rows in particular need a model download to recompute.
    results_path = RESULTS_PATH if not args.only else os.path.join(
        os.path.dirname(RESULTS_PATH), "retrieval_" + "_".join(wanted).replace("+", "-") + ".json")

    documents = datasets.all_documents()
    result = evaluate(documents, retrievers, queries=tuple(QUERIES))
    result["settings"] = {
        "date": datetime.date.today().isoformat(), "k": TOP_K, "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP, "queries": {n: f("METHOD", "/path") for n, f in QUERIES.items()},
        "spec_collection_commit": datasets.COLLECTION_COMMIT, "specs": datasets.SPECS,
        "documents": [{"name": d["name"], "questions": len(d["questions"]), "chunks": len(chunk_spans(d["text"]))}
                      for d in documents],
    }
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    print(markdown_table(result["summary"]))
    print(f"\nFull results and every miss: {os.path.relpath(results_path)}")
    return result


if __name__ == "__main__":
    main()
