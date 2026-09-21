import argparse
import sys
from pathlib import Path

def get_docs_input() -> str:
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
        return path.read_text(encoding="utf-8")

    if args.url:
        import requests
        try:
            response = requests.get(args.url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"ERROR: Could not fetch {args.url}: {e}", file=sys.stderr)
            sys.exit(1)

    return args.text