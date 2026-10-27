# MCP-Forge

[![tests](https://github.com/Priya-Kumari-Chourasia/mcp_forge/actions/workflows/tests.yml/badge.svg)](https://github.com/Priya-Kumari-Chourasia/mcp_forge/actions/workflows/tests.yml)

**Point it at API documentation. Get back a validated MCP (Model Context Protocol) server you can add to Claude Desktop.**

An LLM pipeline built with LangGraph, MCP, A2A and agentic RAG. It reads an OpenAPI spec or prose API docs, generates one MCP tool per endpoint, and sends the code to a separate validator agent over the A2A protocol. The validator calls every tool through a real MCP server against mocked HTTP; its errors are fed back into regeneration until the code passes. The packaged server reads API keys from environment variables and is checked one last time with a real MCP client, the way Claude Desktop connects to it.

---

## Problem Statement

The Model Context Protocol (MCP) is a standard way to expose tools to AI agents. Adopting it has a cost: every team that wants to expose an API has to hand-write the MCP server - read the docs, write a tool function per endpoint, handle authentication and errors, and test it.

MCP-Forge automates that translation. Give it documentation and it produces a packaged MCP server, without a human writing the tool functions.

---

## Architecture

```
                              ORCHESTRATOR            simple or complex? sets the retry budget
                             /            \
                  (OpenAPI spec)          (prose docs)
                       |                       |
                  PARSE DOCS              INDEX DOCS              chunk + embed into ChromaDB
                  (pure code,                  |
                   no LLM)            DISCOVER ENDPOINTS          which endpoints exist?
                       |                       |
                       |                   RETRIEVE <-----------+
                       |                       |                |   AGENTIC RAG LOOP
                       |               GRADE RETRIEVAL -> REWRITE QUERY
                       |                       |
                       +----------+------------+
                                  v
                          GRADE EXTRACTION                drop incomplete endpoints, apply --tags/--paths,
                                  |                       name every tool, settle base URL + auth
                            GENERATE TOOL <------------------------+
                                  |                                |   SELF-CORRECTING LOOP
                                  |  A2A: Agent Card, SendMessage  |   errors + previous code
                                  v                                |   go back to the generator
                           VALIDATOR AGENT ------------------------+
                          (separate process)
                                  |
                           PACKAGE SERVER             server.py + README + Claude Desktop config
                                  |                   + validation report
                          MCP CLIENT CHECK            start server.py over stdio, list its tools
```

The LangGraph graph holds the orchestrator, the extraction nodes and the code generator. The validator is an independent agent in its own process, reached only through the A2A protocol.

**The LLM writes only the tool functions.** Everything that must be right in every server - the base URL, authentication, error handling, tool annotations - comes from a fixed, tested template (`server_template.py`). Each generated tool is a few lines that call `_request()`.

---

## Where Each Technology Is Used

| Technology | What it does here | Where to look |
|---|---|---|
| **LangGraph** | The pipeline is a state graph with two loops and conditional routing | `graph.py`, `state.py` |
| **Agentic RAG** | Prose docs are chunked, embedded and retrieved per endpoint; the LLM grades each retrieval and rewrites the query when the context is insufficient | `rag.py`, `rag_nodes.py` |
| **A2A** | The validator publishes an Agent Card and answers `SendMessage`; the pipeline discovers it and calls it with the official `a2a-sdk` client | `validator_agent.py`, `a2a_client.py` |
| **MCP** | The output is an MCP server built on the `mcp` Python SDK; the validator calls every tool through an MCP server, and an MCP client checks the packaged file | `server_template.py`, `behaviour_check.py` |
| **httpx** | The HTTP client inside every generated server; the validator replaces its send function with a recorder | `server_template.py`, `behaviour_check.py` |
| **Flask** | Hosts the validator agent's HTTP endpoints | `validator_agent.py` |
| **Groq** | LLM backend (`openai/gpt-oss-120b`) | `llm.py` |
| **ChromaDB** | In-memory vector store for the documentation chunks | `rag.py` |

---

## The Generated Server

A run writes a folder to `generated/<api-name>/`:

| File | What it is |
|---|---|
| `server.py` | The MCP server: the template's header, then one tool per endpoint |
| `claude_desktop_config.json` | The entry to paste into Claude Desktop's config, with an empty slot for each credential |
| `README.md` | The tools, the settings, and how to add the server to Claude Desktop |
| `requirements.txt` | `mcp` and `httpx`, pinned |
| `validation_report.json` | Every codegen attempt and its errors, every retrieval round, and the MCP client check |

What makes the tools usable by an AI client, not just runnable:

- **Credentials come from environment variables, never tool arguments.** The spec's security schemes become variables such as `PETSTORE_API_KEY` or `PETSTORE_TOKEN`. A parameter that carries a credential (like Petstore's `api_key` header) is not offered to the AI. For docs that don't declare their authentication, add `--auth bearer` (or `basic`, `header:NAME`, `query:NAME`).
- **Errors say what happened.** An HTTP error reaches the AI as `GET /pet/999 failed with HTTP 404: {"message": "Pet not found"}` instead of a generic `Error executing tool`.
- **Arguments are described.** Each argument carries the spec's description, and allowed values become a `Literal[...]` type, so the client sees `status: one of available, pending, sold`.
- **Tools are marked by what they do.** GET tools are read-only; DELETE, PUT and PATCH are destructive, so clients can ask before running them.
- **Names are stable and readable.** Tool names come from the spec's `operationId` (`findPetsByStatus` -> `find_pets_by_status`), decided in code rather than by the LLM.

### Choosing endpoints

A server with hundreds of tools overwhelms the AI using it, so at most 40 endpoints are kept by default.

```bash
python graph.py --url <spec> --tags pet store              # only endpoints with these OpenAPI tags
python graph.py --url <spec> --paths "/repos/*/issues*"    # only paths matching a pattern
python graph.py --url <spec> --max-tools 60                # raise the cap
```

---

## Agentic RAG (prose docs)

Plain RAG retrieves once and uses whatever comes back. Here the pipeline checks its own retrieval:

1. **index_docs** - the whole document is split into overlapping 1,200-character chunks and embedded into ChromaDB (all-MiniLM-L6-v2).
2. **discover_endpoints** - the document is scanned window by window to list every `METHOD /path`, so nothing is cut off by a context limit.
3. **retrieve** - for one endpoint, fetch the 4 most relevant chunks.
4. **grade_retrieval** - the LLM judges whether that context is enough to describe the endpoint. If yes, it extracts the description, parameters and base URL. If no, it says what is missing and proposes a better query.
5. **rewrite_query** - adopt the better query and retrieve again, up to 3 rounds per endpoint.

```
RAG 3 (retrieve) DELETE /widgets/{id} round 1 -> query: "DELETE /widgets/{id} endpoint description parameters base URL"
  GRADE: insufficient (missing: what the endpoint does and its parameters)
RAG 4 (rewrite_query) -> "removing a widget permanently force flag recycle bin"
RAG 3 (retrieve) DELETE /widgets/{id} round 2 -> query: "removing a widget permanently force flag recycle bin"
  GRADE: sufficient -> endpoint extracted
```

OpenAPI specs skip RAG entirely. A spec is already structured, so `spec_parser.py` reads it in code with zero LLM calls: absolute base URL, path-level and `$ref` parameters, descriptions and allowed values, request bodies (JSON, form data, lists), whether the response is a JSON array or object, security schemes, JSON or YAML, OpenAPI 3 or Swagger 2.

---

## A2A (the validator agent)

`validator_agent.py` implements the A2A JSON-RPC binding, protocol version 1.0:

- `GET /.well-known/agent-card.json` - the Agent Card: name, skill, and the URL and binding to call
- `POST /` with method `SendMessage` - takes a Message with two parts (the code as text, the endpoints as data) and returns a Task
- `POST /` with method `GetTask` - returns a finished Task by id

The pipeline (`a2a_client.py`) knows only the agent's address. It fetches the Agent Card first, then sends the message using the official `a2a-sdk` client.

A Task is `COMPLETED` whenever validation ran, even if the code failed it: "the code is wrong" is the agent's answer, not a failure of the agent. The verdict is in the artifact: `{"passed": false, "errors": [...]}`.

Not implemented: streaming, push notifications, task cancellation, authentication.

---

## The Self-Correcting Loop

1. `generate_tool` writes the tools in batches of 15 endpoints.
2. The validator agent returns structured errors - which function, which endpoint, what went wrong.
3. `errors_by_batch` maps each error to the batch that produced it.
4. Only those batches are regenerated, with their errors and their previous code in the prompt.

Without step 4 a retry would ask the model the identical question, and at temperature 0 it would get the identical answer. The orchestrator sets the budget: 2 attempts for a simple API, 4 for a complex one.

### What the validator checks

The code under test is exactly what gets packaged: the template's header followed by the generated tools.

| Layer | Check | Catches |
|---|---|---|
| Static | `ast.parse`, at least one tool, no duplicate names | syntax errors, name clashes between batches |
| MCP registration | every tool is registered on a real `MCPServer` | type hints the SDK cannot build a schema from |
| Behavioural | every tool is called **through the MCP server** with arguments built from its own input schema, while HTTP is mocked and fake responses are shaped like the spec (array or object) | exceptions, undefined names, a return type that rejects a JSON array, wrong HTTP method or path, required query parameters or bodies not sent, documented parameters with no argument, tools that bypass `_request`, endpoints with no tool |
| Packaging | the finished `server.py` is started as its own process and an MCP client connects over stdio and lists the tools | a server that does not start, tools missing from the server |

The behavioural layer (`behaviour_check.py`) replaces the function httpx uses to send a request with a recorder, so each tool can be called for real without touching the API: a generated `delete_pet()` is exercised, and nothing is deleted. It runs in a child process with a timeout and without secret environment variables, so generated code never runs inside the validator and a hanging tool is killed. This is process isolation, not a security sandbox.

---

## Limitations

- Validation checks the request each tool sends; it does not check response handling against the real API.
- One base URL per server. OAuth2 is supported by passing an access token you already have; the server does not run the OAuth flow.
- File uploads (binary request bodies) are not supported.
- The validator's child process blocks the usual ways to reach the network or start programs, but determined code can get around them, so the public demo accepts only the built-in examples (see `ALLOW_CUSTOM_DOCS` in `app.py`).
- The validator agent implements `SendMessage` and `GetTask` only.

---

## Project Structure

```
mcp-forge/
|-- graph.py                  # LangGraph wiring: nodes, edges, both loops
|-- state.py                  # Shared state schema
|-- nodes.py                  # Orchestrator, OpenAPI path, endpoint selection, codegen loop, packaging
|-- server_template.py        # The fixed part of every generated server: _request, auth, packaging, MCP client check
|-- rag.py                    # Chunking, embeddings, ChromaDB index
|-- rag_nodes.py              # Agentic RAG nodes: index, discover, retrieve, grade, rewrite
|-- spec_parser.py            # Deterministic OpenAPI/Swagger parsing (JSON + YAML)
|-- validator_agent.py        # The validator: an A2A agent on Flask (separate process)
|-- behaviour_check.py        # Calls generated tools through an MCP server against mocked HTTP, in a child process
|-- a2a_client.py             # Agent Card discovery + SendMessage via a2a-sdk
|-- llm.py                    # LLM abstraction - Groq, or the offline stub
|-- cli.py                    # Command-line input: --file, --url or --text, plus endpoint selection and --auth
|-- app.py                    # Streamlit web UI
|-- pipeline_runner.py        # Runs the graph step by step for the UI; recordings; demo switch
|-- demo_runs/                # Recorded runs the UI can replay
|-- examples/                 # A sample prose API guide
|-- tests/                    # pytest suite
|-- requirements.txt
`-- .env.example
```

---

## Getting Started

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Start the validator agent (its own terminal)
```bash
python validator_agent.py
```
Its Agent Card is then at http://localhost:8000/.well-known/agent-card.json

### 3. Run offline, no API key needed
```bash
python graph.py --text "GET https://catfact.ninja/fact - returns a random cat fact"
```
With no `GROQ_API_KEY` set, a stub LLM in `llm.py` is used. It always answers for the cat-fact API, so this run shows the pipeline, not the model.

### 4. Run with a real LLM
1. Get a free API key at console.groq.com
2. Copy `.env.example` to `.env` and add `GROQ_API_KEY`
3. Point it at documentation:
   ```bash
   python graph.py --url https://petstore3.swagger.io/api/v3/openapi.json    # OpenAPI path
   python graph.py --file examples/bookstore_api.md                          # prose path (agentic RAG)
   ```

For APIs with very long documentation on Groq's free tier, set `MCP_FORGE_BATCH_SIZE=8` to send fewer endpoints per request.

### 5. Use the generated server in Claude Desktop
1. Open `generated/<api-name>/claude_desktop_config.json` and copy its entry into Claude Desktop's config (Settings > Developer > Edit Config), inside the existing `"mcpServers"` block.
2. Fill in any credentials in its `"env"` block. An empty value means "not set".
3. Fully quit Claude Desktop and reopen it, then ask it to use the API.

### 6. Run the web UI
```bash
streamlit run app.py
```
The page shows each stage as it runs: the retrieval rounds, the A2A exchange with the validator
agent, every codegen attempt with the errors fed back, and the generated server with the result of
the MCP client check. It starts the validator agent itself, so one command is enough.

- **Recorded run** replays a saved run from `demo_runs/` with no LLM calls.
- **Live run** calls the LLM now. Tick "Save this run as the recorded demo" to create a recording.
- **Demo switch** breaks the first generated tool on purpose (it swaps the tool's HTTP method) so
  the self-correcting loop has an error to fix. It is fault injection for demonstration; the
  validator's error and the repair are real.

For a public deployment set `PUBLIC_DEMO=true`. Live runs are then limited to the built-in examples
and capped per day, because the validator executes LLM-generated code.

### 7. Run the tests
```bash
python -m pytest
```

| Test file | What it proves |
|---|---|
| `test_self_correction.py` | Attempt 1 fails, the validator's error reaches the retry prompt, attempt 2 passes; only the failing batch is regenerated |
| `test_agentic_rag.py` | An endpoint past the first 6,000 characters is found; an insufficient retrieval triggers a rewritten query |
| `test_a2a.py` | Agent Card at the well-known path; the official SDK client can discover and call the agent |
| `test_validator.py` | A return type that rejects a JSON array, required parameters or bodies not sent, undefined names, wrong methods, missing endpoints, tools that bypass `_request`, and tools that cannot be registered on an MCP server are all caught |
| `test_spec_parser.py` | Relative server URLs, path-level and `$ref` parameters, descriptions and enums, response shapes, security schemes, form and list bodies, YAML |
| `test_server_template.py` | A packaged server, run as its own process and called over MCP, returns a JSON array, reports HTTP 404 with its status, rejects invalid values, and sends credentials from environment variables; endpoint selection; the MCP client check |
| `test_pipeline_runner.py` | The web UI's runner reports every step; the demo switch triggers a real correction |

The loop tests use a scripted LLM so they are deterministic.

---

## Author

**Priya Kumari** - M.Tech, Computer Science and Engineering, IIT Jodhpur
