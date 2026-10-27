# Design Decisions

Why MCP-Forge is built the way it is. Each entry says what was chosen, why,
what else was considered, what it costs, and where to see it working (a test
or a file). Where a choice is weaker than it could be, the entry says so.

Contents

1. [The LLM writes only the tool functions](#1-the-llm-writes-only-the-tool-functions)
2. [OpenAPI specs are parsed in code, not by the LLM](#2-openapi-specs-are-parsed-in-code-not-by-the-llm)
3. [LangGraph for the pipeline](#3-langgraph-for-the-pipeline)
4. [Agentic RAG for prose docs](#4-agentic-rag-for-prose-docs)
5. [ChromaDB and MiniLM for retrieval](#5-chromadb-and-minilm-for-retrieval)
6. [The validator is a separate agent, reached over A2A](#6-the-validator-is-a-separate-agent-reached-over-a2a)
7. [FastAPI and the A2A SDK's server, not Flask](#7-fastapi-and-the-a2a-sdks-server-not-flask)
8. [Validation runs on a worker thread](#8-validation-runs-on-a-worker-thread)
9. [A bounded in-memory task store](#9-a-bounded-in-memory-task-store)
10. [Validation calls every tool through a real MCP server, with HTTP mocked](#10-validation-calls-every-tool-through-a-real-mcp-server-with-http-mocked)
11. [The validator's child process is isolation, not a sandbox](#11-the-validators-child-process-is-isolation-not-a-sandbox)
12. [Only the failing batches are regenerated](#12-only-the-failing-batches-are-regenerated)
13. [The official MCP Python SDK, stdio transport, `ToolError`](#13-the-official-mcp-python-sdk-stdio-transport-toolerror)
14. [Credentials come from environment variables](#14-credentials-come-from-environment-variables)
15. [Tool names, annotations and the endpoint cap](#15-tool-names-annotations-and-the-endpoint-cap)
16. [httpx inside generated servers](#16-httpx-inside-generated-servers)
17. [The Try it tab: read-only by default](#17-the-try-it-tab-read-only-by-default)
18. [Groq as the LLM](#18-groq-as-the-llm)
19. [Streamlit for the web UI](#19-streamlit-for-the-web-ui)
20. [Pinned versions and CI](#20-pinned-versions-and-ci)
21. [Known weaknesses](#21-known-weaknesses)

---

## 1. The LLM writes only the tool functions

**Decision.** Every generated `server.py` is a fixed header from
`server_template.py` (base URL, authentication, the `_request()` helper, error
handling) followed by the LLM's tool functions. Each tool is a few lines that
call `_request("GET", "/pet/{id}", ...)`.

**Why.** The parts that must be right in every server are written once, by a
person, and covered by tests (`test_server_template.py`). The LLM only does the
part that changes per API: mapping endpoints to Python functions. Its output is
shorter, so prompts stay within the token limit, and there are fewer places for
it to make a mistake. The validator checks that every tool goes through
`_request` instead of making its own HTTP calls.

**Alternatives.** Letting the LLM write the whole server. That worked in the
first version, but each server handled auth and errors differently, and a bug in
one of them had to be found again in every run.

**Trade-off.** Generated servers all look the same and are limited to what
`_request` supports (one base URL; no file uploads).

## 2. OpenAPI specs are parsed in code, not by the LLM

**Decision.** `spec_parser.py` reads OpenAPI 3 and Swagger 2 (JSON or YAML)
with zero LLM calls: paths, path-level and `$ref` parameters, descriptions,
allowed values, request bodies, response shape and security schemes.

**Why.** A spec is already structured data. Reading it with code is exact,
repeatable and free; asking an LLM to read it again could invent or drop
parameters. The orchestrator also counts a spec's endpoints in code to choose
the retry budget, instead of asking the LLM.

**Alternatives.** Sending the spec through the RAG path like prose docs. Slower,
costs tokens, and less accurate.

**Proof.** `test_spec_parser.py`.

## 3. LangGraph for the pipeline

**Decision.** The pipeline is a LangGraph state graph (`graph.py`) with
conditional routing (spec or prose) and two loops: the retrieval loop and the
self-correcting codegen loop.

**Why.**
- The flow has real cycles with exit conditions (retry until the validator
  passes or the budget runs out). A graph makes those explicit as edges, not
  `while` loops buried in functions.
- All nodes share one typed state (`state.py`), so every step's inputs and
  outputs are visible.
- `graph.stream(stream_mode="updates")` yields one update per node. The web UI
  is built on that (`pipeline_runner.py`), so it can show each stage as it runs.

**Alternatives.**
- Plain Python functions and loops: would work, but the streaming per step and
  the explicit routing would have to be written by hand.
- CrewAI or AutoGen: built around agents talking to each other. This pipeline
  is mostly a fixed sequence with two loops, so that model fits it less well and
  gives less control over each step.

**Trade-off.** One more framework to learn. The recursion limit has to be
raised (`RUN_CONFIG`), because every retrieval round counts as graph steps.

## 4. Agentic RAG for prose docs

**Decision.** For prose docs, the pipeline:
- lists every `METHOD /path` by scanning the whole document window by window
- for each endpoint, retrieves the 4 best chunks
- has the LLM grade whether that context is enough
- if it is not, rewrites the query and retrieves again, up to 3 rounds

**Why.**
- Plain RAG retrieves once and uses whatever comes back. An endpoint whose
  description uses different words than the query would get the wrong context,
  and the generated tool would be wrong.
- Grading makes the failure visible ("missing: what the endpoint does") and
  gives the rewrite something to aim at.
- Scanning in windows means an endpoint far into a long document is still found.

**Alternatives.** Putting the whole document in the prompt. That fails for long
docs on the free tier's token limits, and the model attends less to the middle
of a long prompt.

**Trade-off.** More LLM calls per endpoint. They are capped by
`MAX_RETRIEVAL_ROUNDS` and `MAX_DOC_CHARS`.

**Proof.** `test_agentic_rag.py`: an endpoint past the first 6,000 characters is
found, and an insufficient retrieval triggers a rewritten query.

## 5. ChromaDB and MiniLM for retrieval

**Decision.** Chunks of 1,200 characters (200 overlap) are embedded with
all-MiniLM-L6-v2 and stored in an in-memory ChromaDB collection.

**Why.**
- ChromaDB's default embedding function runs MiniLM through ONNX, so there is
  no PyTorch to install.
- The in-memory client needs no server.
- The same code would work with a persistent store if indexes had to be kept.

**Honest note.** At this size (one document, at most a few hundred chunks), a
NumPy cosine similarity over the vectors would do the same job. ChromaDB is more
than this needs.

**Planned improvement.** Hybrid retrieval: BM25 plus embeddings. Endpoint paths
and parameter names are exact tokens, which keyword search finds better than
embeddings do.

**Tests.** They use a hash-based embedding (`EMBEDDINGS=hash`), so they run
offline and give the same result every time.

## 6. The validator is a separate agent, reached over A2A

**Decision.** Validation is done by `validator_agent.py`, an A2A agent. The
pipeline knows only its URL. It fetches the Agent Card
(`/.well-known/agent-card.json`), then sends a `SendMessage` with the code and
the endpoints, using the official `a2a-sdk` client.

**Why.**
- **A trust boundary.** The validator runs LLM-written code; the pipeline holds
  the Groq key. Keeping them apart now means the validator can later move into
  a locked-down container without changing the pipeline.
- **A standard contract.** The Agent Card says what the agent does and how to
  call it, so any A2A client can use the validator, not just this pipeline. The
  verdict comes back as a structured artifact: `{"passed": ..., "errors": [...]}`.
- **Room to grow.** A second agent (for example a security reviewer) can be
  added the same way, discovered by its own card.

**Alternatives.** A plain function call: simpler and faster in one process. But
validation would share a process with the API key, and there would be no
contract for other callers.

**Trade-off.**
- An HTTP round trip per validation. That costs milliseconds; validation itself
  takes seconds.
- A second service to run. The web UI starts it automatically when nothing
  answers at its address (`ensure_validator_running`).

**Design detail.** A Task is `COMPLETED` even when the code fails validation,
because "the code is wrong" is the agent's answer. `FAILED` means the validator
itself broke.

**Proof.** `test_a2a.py`.

## 7. FastAPI and the A2A SDK's server, not Flask

**Decision.** The validator is a FastAPI app served by uvicorn. The A2A protocol
is handled by the `a2a-sdk` server (`DefaultRequestHandler`, mounted with
`add_a2a_routes_to_fastapi`). The file contains only the agent's logic, in a
`ValidatorExecutor`.

**Why.**
- The first version used Flask with JSON-RPC written by hand. That meant parsing
  requests, routing methods, choosing error codes and storing tasks: protocol
  work that the SDK's maintainers already do and keep in step with the spec.
- The SDK's server side is ASGI (built on Starlette) and async, so it needs an
  ASGI framework. Flask is WSGI and cannot host it.
- With the SDK server, behaviours the spec requires come for free. Example: a
  request without the `A2A-Version: 1.0` header is refused with the standard
  "version not supported" error (-32009), and that is tested.

**Why FastAPI rather than bare Starlette.** The SDK supports both. FastAPI adds:
- an automatic OpenAPI description and the `/docs` page
- an easy place for later endpoints, such as a health check for Docker

It is the only new package: Starlette and uvicorn were already installed,
because `mcp` and `streamlit` depend on them.

**Why uvicorn.** It is the standard ASGI server and was already a dependency.
`serve_in_thread()` binds the socket first, so with port 0 (any free port, used
in tests) the real port is known before the Agent Card is built.

**Trade-off.** Async code is harder to reason about than Flask's synchronous
handlers. See the next entry for the one place that matters.

**Proof.** `test_a2a.py`: the request without a version header, and the `/docs`
page.

## 8. Validation runs on a worker thread

**Decision.** `ValidatorExecutor.execute` calls
`await asyncio.to_thread(validate_code, ...)`.

**Why.**
- `validate_code` waits for a child process, for up to 30 seconds.
- Called directly inside an async function, it would block uvicorn's event loop.
  While one validation ran, the agent could not answer anything, not even a
  request for its Agent Card.
- On a worker thread, the event loop stays free.

**Alternatives.** Rewriting validation with `asyncio.create_subprocess_exec`.
Fully async, but it would turn simple, testable synchronous code into async code
for no extra benefit here.

**Proof.** `test_the_agent_keeps_answering_while_a_validation_runs`: the Agent
Card is fetched in under 1.5 seconds while a 3-second validation runs. This
test was checked by removing `to_thread`: the test then fails, so it really
tests this.

## 9. A bounded in-memory task store

**Decision.** `BoundedTaskStore` wraps the SDK's `InMemoryTaskStore` and keeps
only the newest 500 tasks.

**Why.**
- Finished tasks are kept so that `GetTask` can return them.
- The SDK's in-memory store never forgets anything, so a validator running for
  weeks would keep growing.
- Verdicts are only read right after validation, so forgetting old ones loses
  nothing that is used.

**Alternatives.** The SDK's database-backed store. It would make tasks survive a
restart, which nothing here needs, at the cost of running a database.

**Proof.** `test_old_tasks_are_forgotten_beyond_the_limit`.

## 10. Validation calls every tool through a real MCP server, with HTTP mocked

**Decision.** `behaviour_check.py`:
- registers the generated tools on a real `MCPServer`
- calls each one through `server.call_tool`, with arguments built from its own
  input schema
- replaces httpx's send function with a recorder that returns responses shaped
  like the spec (a JSON array or an object)
- compares the recorded request with the endpoint: method, path, required query
  parameters, body

**Why.**
- Static checks (does it parse, are names unique) miss most real bugs. The bug
  that showed this: a tool annotated to return `dict` passed every static check,
  then failed in Claude Desktop because the API returned a list. Calling the
  tool through MCP with an array-shaped response catches it.
- Mocking means every tool can really run, including `delete_pet`, without
  touching the API.

**Alternatives.** Calling the real API. It would need credentials, could change
or delete real data, and depends on the API being up.

**Trade-off.** The response *handling* is not checked against the real API,
only the request. This is listed in the README's limitations.

**Proof.** `test_validator.py` (15 cases, one per kind of bug).

## 11. The validator's child process is isolation, not a sandbox

**Decision.** Generated code runs in a child process with:
- a 30-second timeout
- no environment variable whose name contains KEY, TOKEN, SECRET or PASSWORD
- sockets and process spawning blocked

**Why.**
- A hanging tool is killed instead of freezing the validator.
- Generated code cannot read the API key.
- Accidental network calls fail.

**What it is not.** Python code that is determined to escape can undo these
blocks. So it is not a security sandbox, and the project says so instead of
claiming one. Because of this, the public demo (`PUBLIC_DEMO=true`) does not
accept custom docs (`ALLOW_CUSTOM_DOCS` defaults to false): only the built-in
examples can be run.

**Planned.** Run the validator in a container with no network and a read-only
filesystem. The A2A boundary (entry 6) is what makes that possible without
changing the pipeline.

## 12. Only the failing batches are regenerated

**Decision.**
- Tools are generated in batches of 15 endpoints (`MCP_FORGE_BATCH_SIZE`).
- The validator's errors are mapped back to the batch that produced them.
- Only those batches are regenerated, with their errors and their previous code
  in the prompt.

**Why.**
- At temperature 0, sending the same prompt again gives the same answer. A retry
  has to include what went wrong to be any different.
- Regenerating only the failing batches saves tokens and does not risk breaking
  tools that already passed.
- Batching keeps each prompt under the free tier's token limit. Descriptions are
  also cut to 200 characters for the same reason.

**Retry budget.** Set by the orchestrator: 2 attempts for a simple API, 4 for a
complex one.

**Proof.** `test_self_correction.py`.

## 13. The official MCP Python SDK, stdio transport, `ToolError`

**Decision.**
- Generated servers use the official `mcp` SDK (`MCPServer`, `@server.tool`).
- They run over stdio.
- `_request` raises `ToolError` with the status code and the response body.

**Why.**
- **The SDK** is maintained alongside the protocol, so servers follow the spec.
  It builds each tool's input schema from the type hints and checks results
  against the return type.
- **stdio** is how Claude Desktop starts local servers: it launches the process
  and talks over its input and output. There is no port to open and no auth to
  set up.
- **`ToolError`** is the exception whose message the SDK passes on to the
  client. Any other exception reaches the AI as a generic "Error executing
  tool". With `ToolError`, the AI sees
  `GET /pet/999 failed with HTTP 404: {"message": "Pet not found"}` and can tell
  the user what happened. This was verified in Claude Desktop.

**Final check.** At packaging, the finished `server.py` is started as its own
process and an MCP client lists its tools over stdio, the same way Claude
Desktop connects.

**Later.** The streamable HTTP transport, for servers hosted remotely.

## 14. Credentials come from environment variables

**Decision.**
- The spec's security schemes become environment variables, such as
  `PETSTORE_API_KEY` or `PETSTORE_TOKEN`.
- Parameters that carry credentials are not offered to the AI as tool
  arguments.
- `claude_desktop_config.json` has an empty slot for each one.

**Why.**
- A credential passed as a tool argument goes through the LLM. It would appear
  in the conversation and could be repeated back to the user.
- MCP clients already pass environment variables to servers (the `env` block in
  Claude Desktop's config), so this is the standard place for them.

**Same rule elsewhere.**
- The validator's child process gets no secrets (entry 11).
- The Try it tab's server process gets only its declared variables (entry 17).

## 15. Tool names, annotations and the endpoint cap

**Tool names.** They come from the spec's `operationId`, converted in code
(`findPetsByStatus` becomes `find_pets_by_status`), not chosen by the LLM.
- Names stay the same every time a server is regenerated.
- Names never clash between batches.

**Annotations.**
- GET tools are marked read-only.
- PUT, PATCH and DELETE are marked destructive.
- POST is marked as neither read-only nor destructive.

Clients use these marks to ask before running a tool. Claude Desktop showed an
approval prompt for `delete_pet`, and the Try it tab offers only read-only tools
by default.

**Endpoint cap.** At most 40 endpoints by default, and they can be chosen with
`--tags`, `--paths` and `--max-tools`.
- Every tool's description goes into the AI's context.
- With hundreds of tools, it picks the wrong one more often and has less room
  left for the conversation.

## 16. httpx inside generated servers

**Decision.** `_request` uses httpx, with a 30-second timeout.

**Why.**
- httpx sends every request through one function, `Client.send`. The validator
  replaces that function with a recorder (entry 10).
- It supports sync and async, if tools become async later.

**Honest note.** `requests` would also work for the HTTP calls. The deciding
reason is the single function to replace. A generated server needs only two
packages: `mcp` and `httpx`.

## 17. The Try it tab: read-only by default

**Decision.** The Try it tab runs an agent loop:
- Groq gets the generated server's tools over MCP, with
  `ChatGroq.bind_tools`.
- It calls tools until it can answer.
- It stops after 6 rounds of tool calls (`MAX_STEPS`).
- Tool results are cut to 6,000 characters.
- Tools that change data are offered only when the user switches them on. On the
  public demo they are always off.
- The server process gets only its declared credential variables plus proxy
  settings, never the Groq key.

**Why.**
- An interviewer, or any user, can test a generated server in the browser
  without installing Claude Desktop.
- The limits make sure a confused model cannot loop forever, overflow its
  context, or change real data on a public page.

**Proof.** `test_try_it.py`.

## 18. Groq as the LLM

**Decision.** `openai/gpt-oss-120b` on Groq, at temperature 0, behind `llm.py`.

**Why.**
- It has a free tier, so anyone can run the project.
- Inference is fast, which matters with many calls per run.
- The model is open-weight and good at writing code and calling tools.
- `ChatGroq` supports tool calling, which the Try it tab needs.
- Temperature 0 makes runs repeatable.

**Trade-off.** The free tier's token-per-minute limits. That is why there are
batches, description trimming and `MCP_FORGE_BATCH_SIZE`.

**Not locked in.** All calls go through `llm.py`, so changing provider changes
one file. With no key set, an offline stub is used, which is how the tests run.

## 19. Streamlit for the web UI

**Decision.** `app.py` is a Streamlit app.

**Why.**
- The whole project is Python, and Streamlit builds a working UI in Python
  alone.
- It can be hosted for free on Streamlit Community Cloud.
- AppTest lets the UI be tested headlessly.

**Alternatives.** React with a FastAPI backend: more control and a better fit
for many users at once, but far more code for a demo UI.

**Trade-off.**
- Streamlit re-runs the script on every interaction.
- Its hosting cannot run containers, which the sandbox in entry 11 would need.

## 20. Pinned versions and CI

**Decision.**
- Every dependency is pinned to an exact version in `requirements.txt`.
- GitHub Actions runs the whole test suite on every push.

**Why.**
- A fresh install (a new laptop, CI, a hosting service) gets the same versions
  the tests passed with.
- This mattered in practice: installing FastAPI upgraded Starlette, which `mcp`
  and `streamlit` also use. Starlette is now pinned on purpose.

**Trade-off.** Upgrades are manual: change a version, reinstall, run the tests.

## 21. Known weaknesses

What is weak today, and why it was left that way:

| Weakness | Why it is acceptable now | Fix |
|---|---|---|
| The validator is not a security sandbox | The public demo runs only the built-in examples | Container with no network (entry 11) |
| The validator has no authentication | It listens on 127.0.0.1 by default, so only this machine can reach it | Add auth before exposing it (`VALIDATOR_HOST=0.0.0.0`) |
| Responses are not checked against the real API | Checking them would mean calling real APIs with real credentials | Optional live check of read-only tools |
| One base URL per server; no OAuth flow; no file uploads | They cover most REST APIs; OAuth works with a token you already have | Extend `_request` |
| ChromaDB is more than this needs | It does no harm and would work with a persistent store | Hybrid BM25 retrieval (entry 5) |
| Only one live run at a time (`RUN_LOCK`) | The RAG index and the demo switch are module-level | Per-run state |
