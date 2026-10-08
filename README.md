# MCP-Forge

**Point it at API documentation. Get back a validated MCP (Model Context Protocol) server.**

A multi-agent pipeline built with LangGraph, MCP, A2A and agentic RAG. It reads an OpenAPI spec or prose API docs, generates one MCP tool per endpoint, sends the code to a separate validator agent over the A2A protocol, and feeds the validator's errors back into regeneration until the code passes.

---

## Problem Statement

The Model Context Protocol (MCP) is a standard way to expose tools to AI agents. Adopting it has a cost: every team that wants to expose an API has to hand-write the MCP server - read the docs, write a tool function per endpoint, and test them.

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
                          GRADE EXTRACTION
                                  |
                            GENERATE TOOL <------------------------+
                                  |                                |   SELF-CORRECTING LOOP
                                  |  A2A: Agent Card, SendMessage  |   errors + previous code
                                  v                                |   go back to the generator
                           VALIDATOR AGENT ------------------------+
                          (separate process)
                                  |
                           PACKAGE SERVER             server.py + README + validation report
```

The LangGraph graph holds the orchestrator, the extraction nodes and the code generator. The validator is an independent agent in its own process, reached only through the A2A protocol.

---

## Where Each Technology Is Used

| Technology | What it does here | Where to look |
|---|---|---|
| **LangGraph** | The pipeline is a state graph with two loops and conditional routing | `graph.py`, `state.py` |
| **Agentic RAG** | Prose docs are chunked, embedded and retrieved per endpoint; the LLM grades each retrieval and rewrites the query when the context is insufficient | `rag.py`, `rag_nodes.py` |
| **A2A** | The validator publishes an Agent Card and answers `SendMessage`; the pipeline discovers it and calls it with the official `a2a-sdk` client | `validator_agent.py`, `a2a_client.py` |
| **MCP** | The output is an MCP server built on the `mcp` Python SDK, one `@server.tool()` per endpoint | `package_as_mcp_server` in `nodes.py` |
| **Flask** | Hosts the validator agent's HTTP endpoints | `validator_agent.py` |
| **Groq** | LLM backend (`openai/gpt-oss-120b`) | `llm.py` |
| **ChromaDB** | In-memory vector store for the documentation chunks | `rag.py` |

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

OpenAPI specs skip RAG entirely. A spec is already structured, so `spec_parser.py` reads it in code with zero LLM calls: absolute base URL, path-level and `$ref` parameters, request bodies, JSON or YAML, OpenAPI 3 or Swagger 2.

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

| Layer | Check | Catches |
|---|---|---|
| Static | `ast.parse`, at least one function, no duplicate names | syntax errors, name clashes between batches |
| Behavioural | every tool is called with dummy arguments while HTTP is mocked | exceptions, undefined names, relative URLs, wrong HTTP method or path, endpoints with no tool |
| MCP | every function is registered on a real `MCPServer` and the tool list is read back | type hints the SDK cannot build a schema from, tools missing from the server |

The behavioural layer (`behaviour_check.py`) replaces the function `requests` uses to send HTTP with a recorder, so each tool can be called for real without touching the API: a generated `delete_pet()` is exercised, and nothing is deleted. It runs in a child process with a timeout, so generated code never runs inside the validator and a hanging tool is killed. This is process isolation, not a security sandbox.

Every run writes `validation_report.json`: each codegen attempt with its errors, and each retrieval round with its query.

---

## Limitations

- Validation checks the request each tool sends (method and URL). It does not check response handling against real data, request-body contents, or authentication.
- Generated tools have no authentication support yet.
- The MCP check registers the tools on a server and lists them; it does not call them through an MCP client over a transport.
- The validator agent implements `SendMessage` and `GetTask` only.

---

## Project Structure

```
mcp-forge/
|-- graph.py                  # LangGraph wiring: nodes, edges, both loops
|-- state.py                  # Shared state schema
|-- nodes.py                  # Orchestrator, OpenAPI path, codegen loop, packaging
|-- rag.py                    # Chunking, embeddings, ChromaDB index
|-- rag_nodes.py              # Agentic RAG nodes: index, discover, retrieve, grade, rewrite
|-- spec_parser.py            # Deterministic OpenAPI/Swagger parsing (JSON + YAML)
|-- validator_agent.py        # The validator: an A2A agent on Flask (separate process)
|-- behaviour_check.py        # Runs generated tools against mocked HTTP, in a child process
|-- a2a_client.py             # Agent Card discovery + SendMessage via a2a-sdk
|-- llm.py                    # LLM abstraction - Groq, or the offline stub
|-- cli.py                    # Command-line input: --file, --url, or --text
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

### 5. Run the web UI
```bash
streamlit run app.py
```
The page shows each stage as it runs: the retrieval rounds, the A2A exchange with the validator
agent, every codegen attempt with the errors fed back, and the generated server. It starts the
validator agent itself, so one command is enough.

- **Recorded run** replays a saved run from `demo_runs/` with no LLM calls.
- **Live run** calls the LLM now. Tick "Save this run as the recorded demo" to create a recording.
- **Demo switch** breaks the first generated tool on purpose (it strips the host from one URL) so
  the self-correcting loop has an error to fix. It is fault injection for demonstration; the
  validator's error and the repair are real.

For a public deployment set `PUBLIC_DEMO=true`. Live runs are then limited to the built-in examples
and capped per day, because the validator executes LLM-generated code.

### 6. Run the tests
```bash
python -m pytest
```

| Test file | What it proves |
|---|---|
| `test_self_correction.py` | Attempt 1 fails, the validator's error reaches the retry prompt, attempt 2 passes; only the failing batch is regenerated |
| `test_agentic_rag.py` | An endpoint past the first 6,000 characters is found; an insufficient retrieval triggers a rewritten query |
| `test_a2a.py` | Agent Card at the well-known path; the official SDK client can discover and call the agent |
| `test_validator.py` | Relative URLs, undefined names, wrong methods, missing endpoints and tools that cannot be registered on an MCP server are all caught |
| `test_spec_parser.py` | Relative server URLs, path-level and `$ref` parameters, request bodies, YAML |

The loop tests use a scripted LLM so they are deterministic.

---

## Author

**Priya Kumari** - M.Tech, Computer Science and Engineering, IIT Jodhpur

