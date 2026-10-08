import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


class _TextExtractor(HTMLParser):
    """Collects the readable text of an HTML page, skipping code and markup."""
    SKIP = {"script", "style", "noscript", "svg", "head", "template", "nav", "footer"}
    BLOCK = {"p", "div", "br", "li", "tr", "table", "section", "article", "pre",
             "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "dt", "dd"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    text = "".join(extractor.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)            # collapse runs of spaces
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()        # at most one blank line in a row


def prepare_docs(text: str) -> str:
    """A docs URL usually returns a web PAGE: HTML full of scripts, styles and
    menus. Feeding that to the pipeline wastes most of the LLM calls on markup,
    so a page is reduced to its readable text first. Specs and plain text are
    returned unchanged."""
    head = text[:2000].lower()
    if "<html" in head or "<!doctype html" in head:
        cleaned = html_to_text(text)
        print(f"[input] HTML page detected: {len(text)} characters reduced to {len(cleaned)} of readable text")
        return cleaned
    return text


def get_docs_input() -> tuple:
    """Returns (docs_text, source_url). source_url is None unless --url was used."""
    parser = argparse.ArgumentParser(description="MCP-Forge: turn API docs into a tested MCP server")
    parser.add_argument("--file", help="Path to a local file with API docs or an OpenAPI spec")
    parser.add_argument("--url", help="URL to fetch API docs or an OpenAPI spec from")
    parser.add_argument("--text", help="Raw API docs text, passed directly")
    args = parser.parse_args()

    sources_given = sum(bool(x) for x in [args.file, args.url, args.text])
    if sources_given == 0:
        parser.error("You must provide one of: --file, --url, or --text")
    if sources_given > 1:
        parser.error("Provide only ONE of --file, --url, or --text, not multiple")

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"ERROR: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        return prepare_docs(path.read_text(encoding="utf-8")), None

    if args.url:
        import requests
        try:
            response = requests.get(args.url, timeout=10)
            response.raise_for_status()
            return prepare_docs(response.text), args.url
        except requests.RequestException as e:
            print(f"ERROR: Could not fetch {args.url}: {e}", file=sys.stderr)
            sys.exit(1)

    return args.text, None