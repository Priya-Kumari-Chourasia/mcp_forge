"""
EVALUATION DATASETS - prose API docs where the right answer is known.

Retrieval is evaluated with one question per endpoint: "which chunk describes
METHOD /path?". To score an answer we must know which chunks are right, so
every question carries the position of the sentence that describes its
endpoint (its "key"). A chunk is relevant if it contains that sentence whole.

Two sources:

1. HAND-WRITTEN - the two Bookstore guides in examples/, labelled by hand.
   Realistic, but small (5 endpoints each).

2. GENERATED - real OpenAPI specs from the APIs.guru collection, rendered
   into prose in two styles. The labels are exact, because the renderer knows
   where it wrote each sentence.
     reference  each endpoint's section shows its METHOD and path, as most API
                references do
     narrative  each endpoint is described in a paragraph WITHOUT its path; the
                paths appear only in an endpoint table at the end (like
                "Withdrawing a title" in the Bookstore guide). The hard case: it
                is why the pipeline grades its retrieval and rewrites the query.
   Honest limit: generated prose is more regular than prose written by people,
   which is why the hand-written guides are measured too.

Every key is at most MAX_KEY_CHARS long. The chunker overlaps chunks by 200
characters, so any text that short lies whole inside at least one chunk: no
question is unanswerable by construction.

The specs are downloaded from one fixed commit of the collection (so everyone
gets the same files) and cached in evaluation/.cache/. They are not copied into
this repository: APIs.guru takes many of them from public sources under fair use.
"""

import os
import re
import urllib.request

from security_rules import strip_invisible
from spec_parser import extract_endpoints_from_openapi, get_api_meta, is_openapi_spec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE_DIR = os.path.join(HERE, ".cache")

COLLECTION_COMMIT = "f04b8d0bcd39c52e1cf3ad7a5fe744709832ae49"     # APIs-guru/openapi-directory, 2026-10-28
COLLECTION_URL = f"https://raw.githubusercontent.com/APIs-guru/openapi-directory/{COLLECTION_COMMIT}/APIs/"

# Specs with enough well-described endpoints for retrieval to have something to choose between.
SPECS = {
    "apis-guru": "apis.guru/2.2.0/openapi.yaml",
    "nytimes-books": "nytimes.com/books_api/3.0.0/openapi.yaml",
    "circleci": "circleci.com/v1/openapi.yaml",
    "httpbin": "httpbin.org/0.9.2/openapi.yaml",
    "spotify": "spotify.com/1.0.0/openapi.yaml",
    "slack": "slack.com/1.7.0/openapi.yaml",
}

MAX_KEY_CHARS = 180

# The sentence that describes each endpoint in the hand-written guides.
HAND_LABELS = {
    "bookstore_api.md": {
        ("GET", "/books"): "To list books, call GET /books.",
        ("GET", "/books/{book_id}"): "To fetch a single book, call GET /books/{book_id}.",
        ("GET", "/books/{book_id}/reviews"): "GET /books/{book_id}/reviews returns every review for one",
        ("POST", "/books"): "POST /books creates a new book.",
        ("DELETE", "/books/{book_id}"): "when a book is no longer sold it can be removed from the catalogue",
    },
    "bookstore_api_long.md": {
        ("GET", "/books"): "To list books, call GET /books.",
        ("GET", "/books/{book_id}"): "To fetch a single book, call GET /books/{book_id}.",
        ("GET", "/books/{book_id}/reviews"): "GET /books/{book_id}/reviews returns every review",
        ("POST", "/books"): "POST /books creates a new book.",
        ("DELETE", "/books/{book_id}"): "When a book is no longer sold it can be taken out of the catalogue for good.",
    },
}


def _document(name, source, style, text, questions):
    """questions: [{"method", "path", "key": (start, end)}], positions in text."""
    return {"name": name, "source": source, "style": style, "text": text, "questions": questions}


# ----------------------------------------------------------------------
# Hand-written guides
# ----------------------------------------------------------------------

def hand_written_documents():
    documents = []
    for filename, labels in HAND_LABELS.items():
        with open(os.path.join(ROOT, "examples", filename), encoding="utf-8") as f:
            text = f.read()
        questions = []
        for (method, path), sentence in labels.items():
            assert text.count(sentence) == 1, f"label not found exactly once in {filename}: {sentence!r}"
            assert len(sentence) <= MAX_KEY_CHARS
            start = text.index(sentence)
            questions.append({"method": method, "path": path, "key": (start, start + len(sentence))})
        documents.append(_document(filename, "hand-written", "hand-written", text, questions))
    return documents


# ----------------------------------------------------------------------
# Generated from real specs
# ----------------------------------------------------------------------

def fetch_spec(name: str) -> str:
    """The spec's text, downloaded once from the pinned commit and cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = os.path.join(CACHE_DIR, name + ".yaml")
    if not os.path.exists(cached):
        with urllib.request.urlopen(COLLECTION_URL + SPECS[name], timeout=60) as response:
            data = response.read().decode("utf-8")
        with open(cached, "w", encoding="utf-8", newline="") as f:      # byte for byte, also on Windows
            f.write(data)
    with open(cached, encoding="utf-8") as f:
        return f.read()


def _clean(text: str) -> str:
    """One line of plain text: invisible characters removed, whitespace collapsed,
    and markdown links reduced to their text."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", strip_invisible(text or ""))
    return " ".join(text.split())


def _key_sentence(endpoint: dict) -> str:
    """The sentence that describes the endpoint: its first sentence, at most
    MAX_KEY_CHARS long. It never contains the path, so the narrative style can
    describe an endpoint without naming it."""
    description = _clean(endpoint.get("description"))
    if not description or description == f"{endpoint['method']} {endpoint['path']}":
        # The spec has no summary: say what the endpoint works on, in words.
        words = [w for w in re.split(r"[/_\-{}.]+", endpoint["path"]) if w]
        description = f"Works with the {' '.join(words) or 'root'} resource"
    sentence = re.split(r"(?<=[.!?])\s", description)[0].rstrip()
    if len(sentence) > MAX_KEY_CHARS - 1:
        sentence = sentence[:MAX_KEY_CHARS - 4].rsplit(" ", 1)[0] + "..."
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence


def _parameter_sentences(endpoint: dict, style: str) -> list:
    lines = []
    for p in endpoint.get("parameters") or []:
        where = p.get("in", "query")
        needed = "required" if p.get("required") else "optional"
        detail = _clean(p.get("description"))[:150].rstrip(".")
        if style == "reference":
            lines.append(f"- `{p['name']}` ({where}, {needed}){': ' + detail if detail else ''}")
        else:
            lines.append(f"It takes the {needed} {where} parameter `{p['name']}`{': ' + detail if detail else ''}.")
    return lines


class _Writer:
    """Builds the document text and remembers where each key sentence went."""

    def __init__(self):
        self.parts, self.length = [], 0

    def add(self, text: str) -> tuple:
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        return start, self.length

    def text(self) -> str:
        return "".join(self.parts)


def _heading(endpoint: dict) -> str:
    """A reference section's heading: the operation's name in words, as many API
    references use ("listUserRepos" -> "List user repos")."""
    name = endpoint.get("operation_id") or f"{endpoint['method'].lower()} {endpoint['path']}"
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    words = " ".join(w for w in re.split(r"[^A-Za-z0-9]+", words) if w).lower()
    return words[:1].upper() + words[1:]


def render(endpoints: list, title: str, style: str):
    """Writes prose docs for the endpoints. Returns (text, questions)."""
    assert style in ("reference", "narrative")
    by_tag = {}
    for endpoint in endpoints:
        by_tag.setdefault((endpoint.get("tags") or ["Other operations"])[0], []).append(endpoint)

    out, questions = _Writer(), []
    out.add(f"# {title} - {'API reference' if style == 'reference' else 'Developer guide'}\n\n")
    for tag, group in by_tag.items():
        out.add(f"## {_clean(tag)}\n\n")
        for endpoint in group:
            key = _key_sentence(endpoint)
            if style == "reference":
                out.add(f"### {_heading(endpoint)}\n\n`{endpoint['method']} {endpoint['path']}`\n\n")
                start, end = out.add(key)
                params = _parameter_sentences(endpoint, style)
                out.add("\n\n" + ("Parameters:\n" + "\n".join(params) + "\n\n" if params else ""))
            else:
                start, end = out.add(key)
                out.add(f" It is a {endpoint['method']} request. " + " ".join(_parameter_sentences(endpoint, style)) + "\n\n")
            questions.append({"method": endpoint["method"], "path": endpoint["path"], "key": (start, end)})

    if style == "narrative":
        out.add("## Endpoint table\n\n| Method | Path |\n|---|---|\n")
        out.add("".join(f"| {e['method']} | {e['path']} |\n" for e in endpoints))
    return out.text(), questions


def spec_documents(names=None):
    """Both renderings of each spec in SPECS (downloaded on first use)."""
    documents = []
    for name in names or SPECS:
        spec = is_openapi_spec(fetch_spec(name))
        endpoints = extract_endpoints_from_openapi(spec, None)
        title = _clean(get_api_meta(spec).get("title")) or name
        for style in ("reference", "narrative"):
            text, questions = render(endpoints, title, style)
            documents.append(_document(f"{name} ({style})", name, style, text, questions))
    return documents


def all_documents():
    return hand_written_documents() + spec_documents()
