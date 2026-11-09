# Evaluation

How well MCP-Forge's parts work, measured rather than claimed. Each section
says what was measured, on what data, the numbers, and what they do and do not
show.

1. [Retrieval quality](#1-retrieval-quality) - does retrieval find the text that describes an endpoint?
2. End-to-end benchmark - does the whole pipeline produce working servers? *(next)*

---

## 1. Retrieval quality

**Result:** measuring 8 ways of retrieving changed how the pipeline retrieves.
The first retrieval round now finds the right text for **91.3%** of endpoints,
up from **82.5%** (66 more of 750 questions), and offline it went from **62.9%**
to **91.2%**.

### What is measured

For prose docs, the pipeline retrieves the 4 chunks that best match a query for
each endpoint, and an LLM then grades whether they are enough
([DECISIONS.md, entry 4](DECISIONS.md#4-agentic-rag-for-prose-docs)). This
measures the **first retrieval round only**: the pipeline's own chunker
(`rag.chunk_spans`), its own first-round query (`rag_nodes.first_round_query`)
and its own ranking code (`rag.BM25`, `rag.reciprocal_rank_fusion`), with no
LLM, no grading and no query rewriting. It shows how often retrieval gets it
right on its own, and so how much is left for the agentic part to fix.

**One question per endpoint:** "which chunk describes `METHOD /path`?" A chunk
counts as relevant if it contains, whole, the sentence that describes the
endpoint. Every such sentence is at most 180 characters and chunks overlap by
200, so every question has at least one relevant chunk (a test checks this).

### Data: 750 questions

| Source | Documents | Questions | Chunks per doc |
|---|---|---|---|
| Hand-written: the two Bookstore guides in `examples/`, labelled by hand | 2 | 10 | 2-7 |
| Generated: 6 real OpenAPI specs from [APIs.guru](https://github.com/APIs-guru/openapi-directory) (APIs.guru, NYT Books, CircleCI, httpbin, Spotify, Slack), each rendered in two styles | 12 | 740 | 1-73 |

The two generated styles:
- **reference**: each endpoint's section shows `METHOD /path`, as most API references do.
- **narrative**: each endpoint is described in a paragraph *without* its path;
  the paths appear only in an endpoint table at the end. This is the hard case,
  like "Withdrawing a title" in the Bookstore guide.

The specs come from one fixed commit of the collection
(`evaluation/datasets.py: COLLECTION_COMMIT`), so the data is the same for everyone.

### Metrics (k = 4, the pipeline's `TOP_K`)

| Metric | Meaning |
|---|---|
| Hit@1 | the top-ranked chunk is relevant |
| **Hit@4** | **at least one relevant chunk is in the top 4: what the grader needs** |
| Recall@4 | the share of relevant chunks in the top 4 (a sentence can sit in two overlapping chunks) |
| MRR | mean of 1 / rank of the first relevant chunk |

`random` is the exact expected score of a random ranking: the floor to beat.

### Results

Two queries were compared for every retriever: **path-only** (`METHOD /path`) and
**generic-words** (`METHOD /path endpoint description parameters base URL`, what
the pipeline sent before this evaluation).

| Retriever / query | Hit@1 | Hit@4 | Recall@4 | MRR | Hit@4 hand-written | Hit@4 reference | Hit@4 narrative | |
|---|---|---|---|---|---|---|---|---|
| random | 0.073 | 0.218 | 0.198 | 0.179 | 0.814 | 0.227 | 0.192 | |
| hash / generic-words | 0.347 | 0.629 | 0.611 | 0.499 | 0.900 | 0.922 | 0.330 | *offline, before* |
| hash / path-only | 0.364 | 0.633 | 0.615 | 0.511 | 0.900 | 0.946 | 0.314 | |
| bm25 / generic-words | 0.420 | 0.855 | 0.839 | 0.610 | 0.900 | 0.981 | 0.727 | |
| **bm25 / path-only** | 0.468 | **0.912** | 0.901 | 0.667 | 0.900 | 0.986 | 0.838 | ***offline, now*** |
| minilm / generic-words | 0.491 | 0.825 | 0.785 | 0.649 | 0.800 | 0.878 | 0.773 | *with MiniLM, before* |
| minilm / path-only | 0.515 | 0.849 | 0.808 | 0.676 | 0.900 | 0.908 | 0.789 | |
| hybrid / generic-words | **0.577** | 0.891 | 0.862 | **0.726** | 0.900 | 0.946 | 0.835 | |
| **hybrid / path-only** | 0.567 | **0.913** | 0.890 | 0.725 | **1.000** | 0.968 | **0.857** | ***with MiniLM, now*** |

750 questions per row. `hash` is a bag-of-words embedding (the old offline
fallback), `bm25` is keyword ranking, `minilm` is all-MiniLM-L6-v2 sentence
embeddings, and `hybrid` combines the MiniLM and BM25 rankings by reciprocal rank
fusion (k = 60). The MiniLM rows were computed on a Windows laptop's CPU;
another machine should give the same rankings, up to tiny floating-point
differences that can only matter for near ties.

### What the numbers show

1. **Every method beats random by a wide margin**: Hit@4 0.63-0.91 against 0.22.
2. **Keywords and meaning help in different places, and combining them wins.**
   MiniLM alone ranks best at the very top (Hit@1 0.515 against BM25's 0.468),
   while BM25 alone finds more in the top 4 (Hit@4 0.912 against 0.849). Endpoint
   paths and parameter names are exact tokens (`conversations.list`, `petId`)
   that keyword search matches and embeddings blur. Hybrid gets the best of both:
   Hit@1 0.567, Hit@4 0.913, MRR 0.725.
3. **The generic query words hurt.** Dropping "endpoint description parameters
   base URL" raises Hit@4 for every method (BM25 0.855 to 0.912, MiniLM 0.825 to
   0.849, hybrid 0.891 to 0.913). The words match unrelated text: for
   `GET /conversations.list` in the Slack docs, "description" pulled up a chunk
   about `setDescription`. Their one benefit is small and at the very top of the
   ranking: they raise Hit@1 on narrative docs for every method, and hybrid's
   overall Hit@1 from 0.567 to 0.577. Hit@4, what the grader needs, matters more.
4. **Narrative docs are much harder.** Even hybrid's Hit@4 falls from 0.968
   (reference) to 0.857 (narrative). When the path is not in the text, the first
   round often misses, which is the case the pipeline's grade-and-rewrite step exists for.
5. **The hand-written guide shows why hybrid wins.** In the long Bookstore guide,
   `DELETE /books/{book_id}` is described only in words ("Withdrawing a title"):
   every keyword method misses it. MiniLM finds it, but misses `POST /books`,
   whose path is written right there in the text. Only hybrid / path-only gets
   all 10 hand-written questions.
6. **The offline fallback is now BM25 alone.** Combining the hash embedding with
   BM25 scored lower than BM25 alone (Hit@4 0.840 against 0.912), so without
   MiniLM there is nothing to combine. (This row is not in the main table; it is
   reproduced with `--only hash+bm25`.)

### The decision

The rule was fixed **before** the MiniLM numbers existed: the pipeline switches
from `minilm / generic-words` only to a combination that raises Hit@4 over all
750 questions by at least 0.03 (about 22 questions) without lowering Hit@4 on
the hand-written guides.

Three qualified: `bm25 / path-only` (+65 questions), `hybrid / generic-words`
(+49) and `hybrid / path-only` (+66). The rule did not say how to choose between
several, which is a gap in it. The choice was the highest Hit@4,
**`hybrid / path-only`**, and the other measures agree with it: against
`bm25 / path-only`, which is one question behind on Hit@4, it is clearly better
on Hit@1 (0.567 vs 0.468), MRR (0.725 vs 0.667) and the hand-written guides
(10/10 vs 9/10).

What the pipeline does now (`rag.py`):

| When | Retrieval | Hit@4 before -> now |
|---|---|---|
| with MiniLM (the default with a real LLM) | MiniLM and BM25 rankings fused by RRF, query `METHOD /path` | 0.825 -> **0.913** |
| without it (offline, the tests) | BM25, query `METHOD /path` | 0.629 -> **0.912** |

**The pipeline runs what was measured.** The evaluation imports its ranking code
from `rag.py`, and `rag.DocIndex.search` was checked against it: the same top 4,
in the same order, for all 750 questions in both modes (in hybrid mode with a
stand-in for MiniLM, which cannot be downloaded in the test environment).
`test_the_pipeline_searches_exactly_as_measured` keeps it that way; it was checked
to fail if the pipeline skips the fusion or ignores the query.

### Limits

- **The choice and the reported score use the same 750 questions.** Picking the
  best of 8 combinations and reporting its score on the same data makes that
  score slightly optimistic. The margin over the old setup (+66 questions) is far
  larger than that effect, but a held-out set of new APIs would give an unbiased number.
- **Generated prose is more regular than prose written by people.** That is why
  the hand-written guides are reported separately, but they are only 10 questions.
- **First round only.** It does not measure the grader or the rewritten queries;
  the end-to-end benchmark covers the whole loop.
- **Exact ranking.** The evaluation ranks every chunk exactly. The pipeline asks
  ChromaDB's approximate (HNSW) index for the full MiniLM ranking; they gave the
  same results here (above), but an approximate index can differ on much larger documents.
- **One document is longer than the pipeline reads.** The pipeline cuts prose docs
  at 60,000 characters (`rag_nodes.MAX_DOC_CHARS`). The Slack narrative document is
  60,878, so the pipeline would lose the last few rows of its endpoint table. The
  evaluation uses the whole document; every key sentence lies before the cut, so this
  affects only the ranking of the table chunks in that one document.

### Reproduce

```bash
python -m evaluation.retrieval_eval                 # all methods; MiniLM downloads its model once (a few minutes on a CPU)
python -m evaluation.retrieval_eval --only bm25,hash
python -m evaluation.retrieval_eval --only hash+bm25   # the 0.840 in point 6 above
```
A full run writes `evaluation/results/retrieval.json`, including every miss for error
analysis. A run with `--only` writes its own file (for example
`evaluation/results/retrieval_bm25_hash.json`), so it never overwrites the full results.
The metric code is checked in `tests/test_evaluation.py`, for example the random
baseline against brute force over every possible ranking.
