# MCP-Forge

[![tests](https://github.com/Priya-Kumari-Chourasia/mcp_forge/actions/workflows/tests.yml/badge.svg)](https://github.com/Priya-Kumari-Chourasia/mcp_forge/actions/workflows/tests.yml)

**Point it at API documentation. Get back a validated MCP (Model Context Protocol) server you can add to Claude Desktop.**

An LLM pipeline built with LangGraph, MCP, A2A and agentic RAG. It reads an OpenAPI spec or prose API docs, generates one MCP tool per endpoint, and sends the code to two separate agents over the A2A protocol. A security reviewer reads the code without running it and refuses anything a tool has no need for; only then does a validator call every tool through a real MCP server against mocked HTTP. Their errors are fed back into regeneration until the code passes. The packaged server reads API keys from environment variables and is checked one last time with a real MCP client, the way Claude Desktop connects to it.

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
                       SECURITY REVIEWER AGENT --------------------+   reads the code, never runs it
                                  |                                |
                                  v                                |
                           VALIDATOR AGENT ------------------------+   runs it against mocked HTTP
                                  |
                       (review never passes: report only, nothing packaged or run)
                                  |
                           PACKAGE SERVER             server.py + README + Claude Desktop config
                                  |                   + validation report
                          MCP CLIENT CHECK            start server.py over stdio, list its tools
```

The LangGraph graph holds the orchestrator, the extraction nodes and the code generator. The security reviewer and the validator are independent agents reached only through the A2A protocol, over HTTP: started with `python security_agent.py` and `python validator_agent.py` each is its own process, and the web UI starts the same apps on background threads when nothing is listening at their addresses.

Why each technology was chosen, and what else was considered, is in [docs/DECISIONS.md](docs/DECISIONS.md).

**The LLM writes only the tool functions.** Everything that must be right in every server - the base URL, authentication, error handling, tool annotations - comes from a fixed, tested template (`server_template.py`). Each generated tool is a few lines that call `_request()`.

---

## Where Each Technology Is Used

| Technology | What it does here | Where to look |
|---|---|---|
| **LangGraph** | The pipeline is a state graph with two loops and conditional routing | `graph.py`, `state.py` |
| **Agentic RAG** | Prose docs are chunked, embedded and retrieved per endpoint; the LLM grades each retrieval and rewrites the query when the context is insufficient | `rag.py`, `rag_nodes.py` |
| **A2A** | Two agents (security reviewer, validator) publish Agent Cards and answer `SendMessage`; the pipeline discovers them and calls them with the official `a2a-sdk` client | `security_agent.py`, `validator_agent.py`, `a2a_server.py`, `a2a_client.py` |
| **Static analysis (`ast`)** | The security reviewer reads the generated code without running it: allowlisted names only, one fixed request per tool | `security_rules.py` |
| **MCP** | The output is an MCP server built on the `mcp` Python SDK; the validator calls every tool through an MCP server, and an MCP client checks the packaged file | `server_template.py`, `behaviour_check.py` |
| **httpx** | The HTTP client inside every generated server; the validator replaces its send function with a recorder | `server_template.py`, `behaviour_check.py` |
| **FastAPI + uvicorn** | Host both agents. The `a2a-sdk` server mounts the A2A routes on each FastAPI app; FastAPI adds the `/docs` page | `a2a_server.py` |
| **Groq** | LLM backend (`openai/gpt-oss-120b`): extraction, code generation, and the tool-calling chat in the Try it tab | `llm.py`, `try_it.py` |
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
- **Path values cannot redirect a tool.** Tools pass the documented path and its values separately (`_request("GET", "/user/{username}", path_params={"username": name})`), and `_request` encodes each value as one path segment. A value such as `../pet/1` would otherwise send the request to a different endpoint than the tool's, including ones left out with `--tags`/`--paths`.
- **Errors say what happened.** An HTTP error reaches the AI as `GET /pet/999 failed with HTTP 404: {"message": "Pet not found"}` instead of a generic `Error executing tool`.
- **Arguments are described.** Each argument carries the spec's description, and allowed values become a `Literal[...]` type, so the client sees `status: one of available, pending, sold`.
- **Tools are marked by what they do.** GET tools are read-only; DELETE, PUT and PATCH are destructive, so clients can ask before running them.
- **Names are stable and readable.** Tool names come from the spec's `operationId` (`findPetsByStatus` -> `find_pets_by_status`), decided in code rather than by the LLM.

### Try it without installing anything

The web UI's **Try it** tab chats with the generated server: ask *"Find available pets"*, and an LLM (Groq) answers by calling the server's tools over MCP, the same way Claude Desktop does, while the real API responds. Each answer shows which tools were called. Only read-only tools are offered unless you switch on "Allow tools that change data" (always off on the public demo). The same chat works from the command line:

```bash
python try_it.py generated/swagger-petstore-openapi-3-0/server.py "Find available pets and show me the first one"
```

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

## A2A (the two agents)

Both agents are FastAPI apps built the same way (`a2a_server.py`). The A2A protocol itself (JSON-RPC binding, protocol version 1.0) is handled by the official `a2a-sdk` server; each agent file holds only its Agent Card and its check: `security_rules.review` for the security reviewer, `validate_code` for the validator.

- `GET /.well-known/agent-card.json` - the Agent Card: name, skill, and the URL and binding to call
- `POST /` with method `SendMessage` - takes a Message with two parts (the code as text, the endpoints as data) and returns a Task
- `POST /` with method `GetTask` - returns a finished Task by id (the newest 500 are kept)
- `GET /docs` - FastAPI's page describing these endpoints

Requests must carry the header `A2A-Version: 1.0` (the SDK client sends it); without it an agent answers with the protocol's "version not supported" error. Each check runs on a worker thread, so an agent keeps answering while a check is running.

The pipeline (`a2a_client.py`) knows only each agent's address. It fetches the Agent Card first, then sends the message using the official `a2a-sdk` client. The security reviewer is sent only the generated tools (the header is the fixed template); the validator is sent the tools under the header, exactly as they will be packaged.

A Task is `COMPLETED` whenever the check ran, even if the code failed it: "the code is wrong" is the agent's answer, not a failure of the agent. The verdict is in the artifact: `{"passed": false, "errors": [...]}`, plus `"warnings"` from the reviewer.

Not implemented: streaming, push notifications, task cancellation, authentication.

---

## The Security Review (the second agent)

`security_agent.py` reviews the generated tools **before** the validator runs them, because the validator executes the code and its child process is isolation, not a sandbox. The reviewer never runs anything: it reads the code with Python's `ast` module (`security_rules.py`). It contains no LLM, so text in the docs cannot talk it into approving something.

| Rule | Refuses | Why |
|---|---|---|
| Request | anything but exactly one `_request` call whose method and path are plain strings equal to the tool's endpoint | the endpoint a tool reaches is then visible without running it, and a tool labelled read-only (from its documented GET) cannot send a DELETE |
| Names | any name except the tool's arguments, its own variables, `_request` and the type names (an allowlist) | no `os`, `open`, `eval`, `getattr`, `print` (stdout is the MCP connection), sockets or imports - including dangers nobody listed in advance |
| Attributes | `__dunder__` attributes and `str.format` | the usual ways out of restrictions in Python |
| Structure | top-level statements, helpers, decorators, redefining `_request`, `server` or other header names | code at the top level runs as soon as the server starts |
| Hidden text | invisible Unicode characters (zero-width, bidirectional controls, tag characters) | text an LLM reads but a person reviewing cannot see; they are also removed from descriptions before generation |

It also **warns**, without blocking, about text in the docs or the tool descriptions that reads like instructions to the AI ("ignore previous instructions", "before using this tool, read ~/.ssh/..."). An MCP client shows tool descriptions to its LLM, so this is how *tool poisoning* works. The phrases are a heuristic, so a person judges them: they are shown in the UI and listed at the top of the generated server's README.

Its findings go back to the generator like the validator's. If the review never passes, the run **fails closed**: nothing is run, no `server.py` is written (any from an earlier run is removed), and the rejected code is kept as `rejected_tools.txt` with the report. An unreachable reviewer also means nothing runs.

---

## The Self-Correcting Loop

1. `generate_tool` writes the tools in batches of 15 endpoints.
2. The security reviewer, then the validator, return structured errors - which function, which endpoint, what went wrong.
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
- Without Docker, the security review and the validator's child process raise the bar a long way, but together they are still not a sandbox (a rule nobody thought of, or a bug in Python itself, is a way out). With Docker Compose the agents run in containers with no internet route and no secrets. Even then, the app container runs the final MCP client check and the Try it tab on code both agents passed, so the public demo accepts only the built-in examples (see `ALLOW_CUSTOM_DOCS` in `app.py`).
- The validator runs whatever it is sent: it relies on its caller (the pipeline) having run the security review first. In Docker nothing outside can reach it; without Docker it listens only on 127.0.0.1.
- The instruction-like text warnings are a heuristic: they can miss a reworded attack and can flag an honest sentence.
- Both agents implement `SendMessage` and `GetTask` only.

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
|-- security_agent.py         # The security reviewer: an A2A agent that reads the code and never runs it
|-- security_rules.py         # Its rules: allowlist, one fixed request per tool, hidden text, tool-poisoning warnings
|-- validator_agent.py        # The validator: an A2A agent that runs the tools against mocked HTTP
|-- a2a_server.py             # What both agents share: FastAPI + a2a-sdk server, task store, uvicorn
|-- behaviour_check.py        # Calls generated tools through an MCP server against mocked HTTP, in a child process
|-- a2a_client.py             # Agent Card discovery + SendMessage via a2a-sdk (calls either agent)
|-- llm.py                    # LLM abstraction - Groq, or the offline stub
|-- cli.py                    # Command-line input: --file, --url or --text, plus endpoint selection and --auth
|-- try_it.py                 # Chat with a generated server over MCP: the agent loop behind the Try it tab
|-- app.py                    # Streamlit web UI
|-- pipeline_runner.py        # Runs the graph step by step for the UI; recordings; demo switch
|-- demo_runs/                # Recorded runs the UI can replay
|-- examples/                 # A sample prose API guide
|-- tests/                    # pytest suite
|-- docs/DECISIONS.md         # Why each technology and design choice, and the alternatives
|-- Dockerfile                # Two images: "agents" (minimal) and "app"
|-- docker-compose.yml        # app + security + validator; the agents locked down and offline
|-- requirements.txt          # Everything (includes requirements-agents.txt)
|-- requirements-agents.txt   # Only what the two agents need
`-- .env.example
```

---

## Getting Started

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Start the two agents (one terminal each)
```bash
python security_agent.py      # http://localhost:8001
python validator_agent.py     # http://localhost:8000
```
Each one's Agent Card is at `/.well-known/agent-card.json` and its endpoints are described at `/docs`. `SECURITY_AGENT_PORT` and `VALIDATOR_PORT` change the ports; `..._HOST=0.0.0.0` accepts connections from other machines. The web UI (step 6) starts both itself, so this step is only for command-line runs.

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
The page shows each stage as it runs: the retrieval rounds, the A2A exchanges with the security
reviewer and the validator, every codegen attempt with the errors fed back, and the generated server
with the result of the MCP client check. The **Try it** tab then lets you chat with that server (it
needs `GROQ_API_KEY`). It starts both agents itself, so one command is enough.

- **Recorded run** replays a saved run from `demo_runs/` with no LLM calls.
- **Live run** calls the LLM now. Tick "Save this run as the recorded demo" to create a recording.
- **Demo switch** breaks the first generated tool on purpose (it swaps the tool's HTTP method) so
  the self-correcting loop has an error to fix. The security review catches it before any code runs;
  its finding and the repair are real, and the validator then checks the repaired code.

For a public deployment set `PUBLIC_DEMO=true`. Live runs are then limited to the built-in examples
and capped per day, because the validator executes LLM-generated code.

### 7. Or run everything with Docker
```bash
docker compose up --build
```
Then open http://localhost:8501. This starts three containers: the web UI, the security reviewer and the validator. Put `GROQ_API_KEY` in `.env` first; without it the offline stub LLM is used.

The two agents handle LLM-written code, so their containers are locked down:

| Setting | What it prevents |
|---|---|
| internal network: no route to the internet | code that got past every check sending data out or calling a real API |
| no `.env`, no Groq key | anything to steal |
| no published ports | anyone but the app calling the agents |
| read-only filesystem (only `/tmp`, in memory), ordinary user, all capabilities dropped | changing the agent's code or gaining privileges |
| 512 MB memory, 1 CPU, 128 processes | a runaway loop or a fork bomb taking the machine down |

In Docker the app never falls back to running an agent inside itself (`START_LOCAL_AGENTS=false`): if an agent container is down, the run stops. The UI listens on 127.0.0.1 only; on a server, start it with `APP_BIND=0.0.0.0`. A containerised run writes servers whose `claude_desktop_config.json` points inside the container, so for Claude Desktop generate them with a local run (step 4).

### 8. Run the tests
```bash
python -m pytest
```
CI runs these on every push, and a second job builds the Docker images, runs the pipeline through the three containers, and checks that neither agent has a route to the internet.

| Test file | What it proves |
|---|---|
| `test_self_correction.py` | Attempt 1 fails, the validator's error reaches the retry prompt, attempt 2 passes; only the failing batch is regenerated |
| `test_agentic_rag.py` | An endpoint past the first 6,000 characters is found; an insufficient retrieval triggers a rewritten query |
| `test_a2a.py` | Both agents: Agent Card at the well-known path; the official SDK client can discover and call them; a request without the protocol version is refused; an agent keeps answering while a check runs; old tasks are forgotten beyond the limit; the reviewer's findings and warnings arrive structured |
| `test_security_rules.py` | f-string paths, a different method or path than documented, two requests, names outside the allowlist (`os`, `eval`, `open`, `print`, `getattr`, `__import__`, `httpx`), imports, `global`, dunder attributes, `.format`, top-level code, redefining `_request`, decorators, calling `_request` indirectly and invisible characters are all refused; instruction-like text is a warning, not an error |
| `test_security_review.py` | In the pipeline: a finding is fed back and fixed, and the unsafe attempt never reaches the validator; code the review never passes is not run or packaged and an older server.py is removed; an unreachable reviewer fails closed; warnings reach the generated README |
| `test_validator.py` | A return type that rejects a JSON array, required parameters or bodies not sent, undefined names, wrong methods, missing endpoints, tools that bypass `_request`, and tools that cannot be registered on an MCP server are all caught |
| `test_spec_parser.py` | Relative server URLs, path-level and `$ref` parameters, descriptions and enums, response shapes, security schemes, form and list bodies, YAML |
| `test_server_template.py` | A packaged server, run as its own process and called over MCP, returns a JSON array, reports HTTP 404 with its status, rejects invalid values, and sends credentials from environment variables; a path value such as `../pet/1` stays inside its own segment; endpoint selection; the MCP client check |
| `test_pipeline_runner.py` | The web UI's runner reports every step; the demo switch triggers a real correction; with `START_LOCAL_AGENTS=false` an unreachable agent stops the run instead of being started inside the app |
| `test_try_it.py` | The chat calls the server's tools over MCP and answers from the API's data; errors reach the model with their status code; tools that change data are offered only when allowed; the loop stops after a fixed number of calls; the server never sees the Groq key |

The loop tests use a scripted LLM so they are deterministic.

---

## Author

**Priya Kumari** - M.Tech, Computer Science and Engineering, IIT Jodhpur
