"""The retrieval evaluation must measure correctly: the metrics, the random
baseline, the ranking methods and the datasets are each checked against cases
whose answer is known. Offline: no download, no model."""
import itertools
import random

import pytest

import rag
from evaluation import datasets
from evaluation import retrieval_eval as R

SPEC = {
    "openapi": "3.0.0", "info": {"title": "Pet API"}, "servers": [{"url": "https://pets.example"}],
    "paths": {
        "/pets": {"get": {"operationId": "listPets", "summary": "Lists every pet in the shop.", "tags": ["pets"],
                          "parameters": [{"name": "limit", "in": "query", "description": "How many."}]}},
        "/pets/{petId}": {"get": {"operationId": "getPet", "summary": "Returns one pet by its identifier.",
                                  "tags": ["pets"],
                                  "parameters": [{"name": "petId", "in": "path", "required": True}]},
                          "delete": {"operationId": "deletePet", "tags": ["pets"]}},
    },
}


def test_scores_for_a_known_ranking():
    row = R.score_ranking([2, 0, 1, 3, 4], relevant={0}, k=4)
    assert row == {"hit@1": 0.0, "hit@4": 1.0, "recall@4": 1.0, "mrr": 0.5, "first_rank": 2}
    row = R.score_ranking([5, 4, 3, 2, 1, 0], relevant={0, 1}, k=4)
    assert row["hit@4"] == 0.0 and row["recall@4"] == 0.0 and row["mrr"] == pytest.approx(1 / 5)


@pytest.mark.parametrize("n, r", [(1, 1), (5, 1), (5, 2), (6, 3), (7, 2)])
def test_random_baseline_matches_every_possible_ranking(n, r):
    """The exact formula against brute force: the average over all n! orderings."""
    rows = [R.score_ranking(list(order), set(range(r)), k=4) for order in itertools.permutations(range(n))]
    expected = R.expected_random(n, r, k=4)
    for metric in ("hit@1", "hit@4", "recall@4", "mrr"):
        assert expected[metric] == pytest.approx(sum(row[metric] for row in rows) / len(rows))


def test_reciprocal_rank_fusion_rewards_agreement():
    """Each item scores the sum of 1 / (60 + its rank). Second in both rankings
    (2/62 = 0.03226) beats first in one and fourth in the other (1/61 + 1/64 = 0.03202)."""
    assert R.reciprocal_rank_fusion([[0, 1, 2, 3], [3, 1, 2, 0]])[0] == 1
    assert R.reciprocal_rank_fusion([[0, 1, 2], [0, 2, 1]]) == [0, 1, 2]     # a tie goes to the lower index


def test_hash_plus_bm25_is_the_fusion_of_the_two_rankings():
    """The 0.840 quoted in docs/EVALUATION.md comes from this ranker; it must be
    exactly hash and BM25 fused by reciprocal rank fusion."""
    chunks = ["the shop sells pets", "pets are fed daily", "invoices are paid monthly", "the shop opens daily"]
    query = "GET /pets"
    hashed, keyword = R.RANKERS["hash"](chunks)(query), R.RANKERS["bm25"](chunks)(query)
    assert R.RANKERS["hash+bm25"](chunks)(query) == R.reciprocal_rank_fusion([hashed, keyword])
    assert "hash+bm25" not in R.DEFAULT_RETRIEVERS.split(",")       # not part of the main table


def test_evaluate_runs_with_its_default_query():
    """Called without `queries`, evaluate() uses the pipeline's own query (path-only)."""
    result = R.evaluate(datasets.hand_written_documents(), ["bm25"])
    assert set(result["summary"]) == {"bm25 / path-only"}


def test_bm25_ranks_the_chunk_with_the_rare_word_first():
    chunks = ["the shop sells pets", "the shop opens daily", "invoices are paid monthly"]
    assert R.BM25(chunks).scores("pets shop").index(max(R.BM25(chunks).scores("pets shop"))) == 0
    assert R.tokenize("GET /pets/{petId}") == ["get", "pets", "pet", "id"]


def test_chunk_text_is_built_from_chunk_spans():
    text = "".join(random.Random(1).choice("ab \n") for _ in range(5000))
    assert rag.chunk_text(text) == [text[s:e].strip() for s, e in rag.chunk_spans(text)]


def test_any_text_up_to_the_key_length_lies_whole_in_some_chunk():
    """Why no question is unanswerable: chunks overlap enough that every span of
    MAX_KEY_CHARS characters fits inside at least one of them."""
    rnd = random.Random(2)
    for _ in range(20):
        text = "".join(rnd.choice("abcde \n\n") for _ in range(rnd.randint(300, 6000)))
        spans = rag.chunk_spans(text)
        for start in range(0, len(text) - datasets.MAX_KEY_CHARS, 37):
            assert R.relevant_chunks(spans, (start, start + datasets.MAX_KEY_CHARS)), (len(text), start)


def test_rendered_docs_are_labelled_correctly():
    from spec_parser import extract_endpoints_from_openapi
    endpoints = extract_endpoints_from_openapi(SPEC, None)
    for style in ("reference", "narrative"):
        text, questions = datasets.render(endpoints, "Pet API", style)
        assert len(questions) == 3
        keys = [text[s:e] for s, e in (q["key"] for q in questions)]
        assert keys[0] == "Lists every pet in the shop." and all(len(k) <= datasets.MAX_KEY_CHARS for k in keys)
        spans = rag.chunk_spans(text)
        assert all(R.relevant_chunks(spans, q["key"]) for q in questions)
        if style == "narrative":           # the paths appear only in the endpoint table
            body = text.split("## Endpoint table")[0]
            assert "/pets" not in body and "| DELETE | /pets/{petId} |" in text
        else:
            assert "`DELETE /pets/{petId}`" in text and "### Delete pet" in text


def test_hand_written_labels_point_at_the_right_sentences():
    for doc in datasets.hand_written_documents():
        assert len(doc["questions"]) == 5
        withdraw = [q for q in doc["questions"] if q["method"] == "DELETE"][0]
        assert "no longer sold" in doc["text"][withdraw["key"][0]:withdraw["key"][1]]


def test_the_whole_evaluation_runs_with_every_retriever(monkeypatch):
    """MiniLM needs a download, so a stand-in embedding exercises the same code."""
    monkeypatch.setattr(R, "_embed_minilm", R.hash_embed)
    monkeypatch.setitem(R.RANKERS, "minilm", R._cosine_ranker(R.hash_embed))
    from spec_parser import extract_endpoints_from_openapi
    text, questions = datasets.render(extract_endpoints_from_openapi(SPEC, None), "Pet API", "reference")
    docs = datasets.hand_written_documents() + [{"name": "pets", "source": "pets", "style": "reference",
                                                 "text": text, "questions": questions}]
    result = R.evaluate(docs, ["random", "hash", "bm25", "minilm", "hybrid"], queries=("path-only", "generic-words"))
    assert set(result["summary"]) == {"random", "hash / path-only", "hash / generic-words", "bm25 / path-only",
                                      "bm25 / generic-words", "minilm / path-only", "minilm / generic-words",
                                      "hybrid / path-only", "hybrid / generic-words"}
    for run in result["summary"].values():
        assert run["all"]["questions"] == 13
        assert all(0 <= run["all"][m] <= 1 for m in ("hit@1", "hit@4", "recall@4", "mrr"))
    assert "| bm25 / path-only | 13 |" in R.markdown_table(result["summary"])


def test_each_query_is_embedded_once(monkeypatch):
    """With MiniLM every embedding call runs a neural network. Embedding the query
    again for every chunk made a full run take hours instead of minutes."""
    calls = []

    def counting_embed(texts):
        calls.append(len(texts))
        return R.hash_embed(texts)
    monkeypatch.setattr(R, "_query_vectors", {})
    rank = R._cosine_ranker(counting_embed)(["chunk one", "chunk two", "chunk three", "chunk four"])
    for _ in range(3):
        rank("GET /pets")
    rank("GET /pets/{petId}")
    assert calls == [4, 1, 1]          # the chunks in one call, then each distinct query once


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_the_pipeline_searches_exactly_as_measured(mode, monkeypatch):
    """The numbers in docs/EVALUATION.md describe the pipeline only if
    rag.DocIndex.search returns the evaluation's top 4, in the same order. (A
    stand-in embedding replaces MiniLM in both, since the model is a download.)"""
    import rag_nodes
    if mode == "hybrid":
        monkeypatch.setenv("EMBEDDINGS", "minilm")
        monkeypatch.setattr(rag, "_embed_minilm", R.hash_embed)
        monkeypatch.setattr(R, "_embed_minilm", R.hash_embed)
        monkeypatch.setattr(R, "_query_vectors", {})
    else:
        monkeypatch.setenv("EMBEDDINGS", "none")
    for doc in datasets.hand_written_documents() + [_large_document()]:
        index = rag.DocIndex(doc["text"])
        assert (index.collection is not None) == (mode == "hybrid")
        measured = R.RANKERS[mode](index.chunks)
        for q in doc["questions"]:
            query = rag_nodes.first_round_query(q["method"], q["path"])
            assert index.search(query, k=4) == [index.chunks[i] for i in measured(query)[:4]]


def _large_document():
    """40 endpoints made from a fixed seed: enough chunks (the Bookstore guides
    have 2 and 7) for different ranking methods to disagree, so the test above
    fails if the pipeline ranks differently from what was measured."""
    from spec_parser import extract_endpoints_from_openapi
    rnd = random.Random(7)
    nouns = ["invoice", "customer", "order", "shipment", "refund", "coupon", "address", "payment", "basket", "review"]
    verbs = ["Lists", "Returns", "Creates", "Updates", "Cancels", "Archives", "Exports", "Validates"]
    paths = {}
    for i in range(40):
        noun, other = rnd.sample(nouns, 2)
        paths[f"/{noun}s/{{{noun}Id}}/{other}s{i}"] = {"get": {
            "operationId": f"op{i}", "tags": [noun],
            "summary": f"{rnd.choice(verbs)} the {other}s of a {noun} for {rnd.choice(nouns)} reports.",
            "parameters": [{"name": f"{noun}Id", "in": "path", "required": True, "description": f"The {noun}."}]}}
    endpoints = extract_endpoints_from_openapi({"openapi": "3.0.0", "info": {"title": "Shop"}, "paths": paths}, None)
    text, questions = datasets.render(endpoints, "Shop", "narrative")
    assert len(rag.chunk_spans(text)) >= 8
    return {"name": "large", "style": "narrative", "text": text, "questions": questions}
