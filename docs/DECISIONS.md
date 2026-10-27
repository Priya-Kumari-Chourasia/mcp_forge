# Design Decisions

Why MCP-Forge is built the way it is. Each entry says what was chosen, why,
what else was considered, what it costs, and where to see it working (a test
or a file). Where a choice is weaker than it could be, the entry says so.

Contents

Pipeline
1. [The LLM writes only the tool functions](#1-the-llm-writes-only-the-tool-functions)
2. [OpenAPI specs are parsed in code, not by the LLM](#2-openapi-specs-are-parsed-in-code-not-by-the-llm)
3. [LangGraph for the pipeline](#3-langgraph-for-the-pipeline)
4. [Agentic RAG for prose docs](#4-agentic-rag-for-prose-docs)
5. [ChromaDB and MiniLM for retrieval](#5-chromadb-and-minilm-for-retrieval)

Agents
6. [Two separate agents, reached over A2A](#6-two-separate-agents-reached-over-a2a)
7. [FastAPI and the A2A SDK's server, not Flask](#7-fastapi-and-the-a2a-sdks-server-not-flask)
8. [Checks run on a worker thread](#8-checks-run-on-a-worker-thread)
9. [A bounded in-memory task store](#9-a-bounded-in-memory-task-store)

Security
10. [A security review runs before the validator](#10-a-security-review-runs-before-the-validator)
11. [The review is fixed rules, not an LLM](#11-the-review-is-fixed-rules-not-an-llm)
12. [An allowlist, not a blocklist](#12-an-allowlist-not-a-blocklist)
13. [Path values are encoded by the template](#13-path-values-are-encoded-by-the-template)
14. [Tool poisoning: warn, and strip hidden characters](#14-tool-poisoning-warn-and-strip-hidden-characters)
15. [Validation calls every tool through a real MCP server, with HTTP mocked](#15-validation-calls-every-tool-through-a-real-mcp-server-with-http-mocked)
16. [Isolation, not a sandbox](#16-isolation-not-a-sandbox)

Generation and the generated server
17. [Only the failing batches are regenerated](#17-only-the-failing-batches-are-regenerated)
18. [The official MCP Python SDK, stdio transport, `ToolError`](#18-the-official-mcp-python-sdk-stdio-transport-toolerror)
19. [Credentials come from environment variables](#19-credentials-come-from-environment-variables)
20. [Tool names, annotations and the endpoint cap](#20-tool-names-annotations-and-the-endpoint-cap)
21. [httpx inside generated servers](#21-httpx-inside-generated-servers)

Product
22. [The Try it tab: read-only by default](#22-the-try-it-tab-read-only-by-default)
23. [Groq as the LLM](#23-groq-as-the-llm)
24. [Streamlit for the web UI](#24-streamlit-for-the-web-ui)
25. [Pinned versions and CI](#25-pinned-versions-and-ci)
26. [Known weaknesses](#26-known-weaknesses)

---

## 1. The LLM writes only the tool functions

**Decision.** Every generated `server.py` is a fixed header from
`server_template.py` (base URL, authentication, the `_request()` helper, error
handling) followed by the LLM's tool functions. Each tool is a few lines that
call `_request("GET", "/pet/{petId}", path_params={"petId": pet_id})`.

**Why.** The parts that must be right in every server are written once, by a
person, and covered by tests (`test_server_template.py`). The LLM only does the
part that changes per API: mapping endpoints to Python functions. Its output is
shorter, so prompts stay within the token limit, and there are fewer places for
it to make a mistake. It also makes the generated part small and regular enough
for the security review to check it strictly (entries 10-12).

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
- The flow has real cycles with exit conditions (retry until both agents pass
  the code or the budget runs out). A graph makes those explicit as edges, not
  `while` loops buried in functions. The security review's three outcomes
  (validate, retry, rejected) are one conditional edge.
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

## 6. Two separate agents, reached over A2A

**Decision.** The generated code is checked by two A2A agents: the security
reviewer (`security_agent.py`) and the validator (`validator_agent.py`). The
pipeline knows only their URLs. For each, it fetches the Agent Card
(`/.well-known/agent-card.json`), then sends a `SendMessage` with the code and
the endpoints, using the official `a2a-sdk` client.

**Why.**
- **A trust boundary.** The validator runs LLM-written code; the pipeline holds
  the Groq key. Keeping them apart means the validator can later move into a
  locked-down container without changing the pipeline.
- **Two different questions.** "Is it safe to run?" (answered by reading the
  code) and "does it work?" (answered by running it) need different trust: the
  reviewer never runs code, the validator must. Separate agents keep that line
  sharp, and each verdict is reported on its own.
- **A standard contract.** An Agent Card says what the agent does and how to
  call it, so any A2A client can use either agent, not just this pipeline. The
  verdict comes back as a structured artifact: `{"passed": ..., "errors": [...]}`.

**Alternatives.** Plain function calls: simpler and faster in one process. But
the code would be checked in the same process that holds the API key, and there
would be no contract for other callers. Putting the security rules inside the
validator: one agent fewer, but then the component that runs code would also be
the one deciding whether it is safe to run.

**Trade-off.**
- An HTTP round trip per check. That costs milliseconds; validation takes seconds.
- Two more services to run. The web UI starts both automatically when nothing
  answers at their addresses (`ensure_agent_running`).

**Design detail.** A Task is `COMPLETED` even when the code fails the check,
because "the code is wrong" is the agent's answer. `FAILED` means the agent
itself broke.

**Proof.** `test_a2a.py`.

## 7. FastAPI and the A2A SDK's server, not Flask

**Decision.** Each agent is a FastAPI app served by uvicorn. The A2A protocol is
handled by the `a2a-sdk` server (`DefaultRequestHandler`, mounted with
`add_a2a_routes_to_fastapi`). What both agents share lives in `a2a_server.py`;
each agent file holds only its Agent Card and its check.

**Why.**
- The first version used Flask with JSON-RPC written by hand. That meant parsing
  requests, routing methods, choosing error codes and storing tasks: protocol
  work that the SDK's maintainers already do and keep in step with the spec.
- The SDK's server side is ASGI (built on Starlette) and async, so it needs an
  ASGI framework. Flask is WSGI and cannot host it.
- With the SDK server, behaviours the spec requires come for free. Example: a
  request without the `A2A-Version: 1.0` header is refused with the standard
  "version not supported" error (-32009), and that is tested.
- Both agents take code plus endpoints and return a verdict, so one executor
  class (`CheckExecutor`) serves both. Adding the second agent was a card and a
  check function, not a second copy of the server code.

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

## 8. Checks run on a worker thread

**Decision.** `CheckExecutor.execute` calls
`await asyncio.to_thread(self.check, ...)`.

**Why.**
- The validator's check waits for a child process, for up to 30 seconds.
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
- The SDK's in-memory store never forgets anything, so an agent running for
  weeks would keep growing.
- Verdicts are only read right after the check, so forgetting old ones loses
  nothing that is used.

**Alternatives.** The SDK's database-backed store. It would make tasks survive a
restart, which nothing here needs, at the cost of running a database.

**Proof.** `test_old_tasks_are_forgotten_beyond_the_limit`.

## 10. A security review runs before the validator

**Decision.** After code generation, the security reviewer checks the tools
first. Only code it passes is sent to the validator. Its findings go back to the
generator like the validator's errors. If it never passes the code, the run
**fails closed**:
- nothing is run, not even the final MCP client check
- no `server.py` is written, and one from an earlier run of the same API is
  removed (it would not match this run's report)
- the rejected code is kept as `rejected_tools.txt`, a name that cannot be run
  by accident

An unreachable reviewer also stops the run before anything executes.

**Why.** The validator *executes* the generated code, and its child process is
isolation, not a sandbox (entry 16). So the decision "is this safe to run" has
to be made before running it, by something that does not run it.

**Alternatives.** Relying on the validator's child process alone: it limits
damage after the code is already running. Reviewing after validation: too late,
the code has already run.

**Trade-off.** One more A2A call per attempt (milliseconds). A finding uses one
attempt from the retry budget, like a validator error.

**Proof.** `test_security_review.py`:
- a finding is fed back and fixed, and the unsafe attempt never reaches the
  validator
- code that never passes is not run or packaged
- an unreachable reviewer fails closed

The first claim was checked by making the router ignore the review: the tests
then fail.

## 11. The review is fixed rules, not an LLM

**Decision.** `security_rules.py` reads the code with Python's `ast` module and
applies fixed rules. No model is involved.

**Why.**
- **Prompt injection.** The docs being processed are untrusted. If they contain
  "ignore previous instructions", an LLM reviewer reading them alongside the
  code could be told to approve it. Rules cannot be persuaded.
- **Repeatable.** The same code always gets the same verdict, so a test can
  prove each rule (`test_security_rules.py`), and a finding can be explained by
  pointing at the rule.
- **Cheap and fast.** No tokens, milliseconds per review.

**Alternatives.** An LLM reviewer: better at judging intent, but it can be
manipulated and is not repeatable. A general linter such as Bandit: built to
find risky patterns in ordinary code. Here the code has a narrow, known shape
(one request per tool), which allows much stricter rules than a general tool
can apply.

**Trade-off.** Rules only catch what they describe. That is why the rules are
an allowlist (next entry): what is not described as allowed is refused.

## 12. An allowlist, not a blocklist

**Decision.** A tool may use only:
- its own arguments and variables it assigns itself
- `_request`
- the type names
- a few harmless built-ins (`str`, `int`, `len`, ...)

Everything else is refused. On top of that, a tool must make exactly one
`_request` call, with the method and the path as plain strings equal to its
endpoint's.

**Why.**
- A blocklist (no `os`, no `eval`, ...) only stops the dangers someone thought
  of in advance. Python has many indirect routes (`getattr`, `__import__`,
  `().__class__.__subclasses__()`, format strings). An allowlist stops them
  without having to name them. `__dunder__` attributes and `str.format` are
  refused separately, because those routes go through attributes, not names.
- Tool code needs very little, so the allowlist can be small. Even `print` is
  refused: over stdio, a tool's output *is* the MCP connection, so printing
  corrupts it.
- A fixed method and path mean the endpoint a tool reaches is visible without
  running it. The tool's read-only or destructive label is set from its
  documented method (entry 20), so this also guarantees that a tool labelled
  read-only cannot send a DELETE.

**Trade-off.** A valid but unusual tool (one that needs a helper function, for
example) is refused. The code-generation prompt states the same rules, so the
LLM rarely writes such code. When it does, the finding tells it what to change.

**Proof.** `test_security_rules.py`.

## 13. Path values are encoded by the template

**Decision.** Tools pass the documented path and its values separately:
`_request("GET", "/user/{username}", path_params={"username": name})`.
`_request` fills in the placeholders and URL-encodes each value as one path
segment. It refuses `""`, `"."` and `".."`.

**Why.** This fixes a real bug, found while designing the review. Tools used to
build the path themselves with f-strings, so
`get_user_by_name(username="../pet/1")` sent `GET /pet/1`. The URL parser
resolves `..`, so a tool could reach a different endpoint than its own,
including endpoints deliberately left out with `--tags`/`--paths`. The AI
choosing arguments can be steered by text it has read, so this is reachable.
- Encoding `/` as `%2F` keeps a value inside its segment.
- `.` and `..` contain no characters that encoding changes, so they are refused
  outright.

**Why in the template.** One tested implementation for every server, instead of
trusting the LLM to encode correctly in every tool. A literal path is also what
lets the review compare it with the endpoint (entry 12).

**Proof.** `test_a_path_value_cannot_reach_another_endpoint` (checked by removing
the encoding: the test then fails).

## 14. Tool poisoning: warn, and strip hidden characters

**Decision.**
- Invisible Unicode characters are removed from every description before it
  reaches the LLM. This covers zero-width characters, bidirectional controls and
  tag characters (Unicode category Cf).
- In the generated code, they are an error.
- Text that reads like instructions to an AI is a **warning**: it is shown in
  the UI and listed at the top of the generated server's README, but it does
  not block. Examples are "ignore previous instructions" and "before using this
  tool, read ~/.ssh/...".

**Why.**
- An MCP client shows tool descriptions to its LLM, so instructions hidden in
  them reach the AI that uses the server. This is a known MCP attack called
  *tool poisoning*.
- Invisible characters are how such text hides from a person reviewing the
  docs. Since nothing legitimate needs them in an API description, removing
  them costs nothing.
- Instruction-like phrases are a guess. A reworded attack can avoid them, and an
  honest sentence can match. Blocking on a guess would stop honest APIs, so a
  person judges.

**Proof.** `test_security_rules.py`; `test_warnings_reach_the_generated_readme`.

## 15. Validation calls every tool through a real MCP server, with HTTP mocked

**Decision.** `behaviour_check.py`:
- registers the generated tools on a real `MCPServer`
- calls each one through `server.call_tool`, with arguments built from its own
  input schema
- replaces httpx's send function with a recorder that returns responses shaped
  like the spec (a JSON array or an object)
- compares the recorded request with the endpoint: method, path, required query
  parameters, body

**Why.**
- Reading the code (the review) cannot show what it does when it runs. The bug
  that showed this: a tool annotated to return `dict` passed every static check,
  then failed in Claude Desktop because the API returned a list. Calling the
  tool through MCP with an array-shaped response catches it. So does a tool that
  forgets to pass its path value, which looks fine when read
  (`test_self_correction.py` uses exactly that bug).
- Mocking means every tool can really run, including `delete_pet`, without
  touching the API.

**Alternatives.** Calling the real API. It would need credentials, could change
or delete real data, and depends on the API being up.

**Trade-off.** The response *handling* is not checked against the real API,
only the request. This is listed in the README's limitations.

**Proof.** `test_validator.py`.

## 16. Isolation, not a sandbox

**Decision.** Code that passed the review runs in the validator's child process
with:
- a 30-second timeout
- no environment variable whose name contains KEY, TOKEN, SECRET or PASSWORD
- sockets and process spawning blocked

**Why.**
- A hanging tool is killed instead of freezing the validator.
- Generated code cannot read the API key from the environment.
- Accidental network calls fail.

**What it is not.** The review and the child process together raise the bar a
long way, but neither is a sandbox. A rule nobody thought of, or a bug in Python
itself, is still a way out. The project says so instead of claiming otherwise.
Because of this, the public demo (`PUBLIC_DEMO=true`) does not accept custom
docs (`ALLOW_CUSTOM_DOCS` defaults to false): only the built-in examples can be
run.

**Planned.** Run the validator in a container with no network and a read-only
filesystem. The A2A boundary (entry 6) is what makes that possible without
changing the pipeline.

## 17. Only the failing batches are regenerated

**Decision.**
- Tools are generated in batches of 15 endpoints (`MCP_FORGE_BATCH_SIZE`).
- Errors from either agent are mapped back to the batch that produced them.
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
complex one. It is shared by both agents: an attempt counts whether the reviewer
or the validator rejected it.

**Proof.** `test_self_correction.py`.

## 18. The official MCP Python SDK, stdio transport, `ToolError`

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

## 19. Credentials come from environment variables

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
- Only the template reads credentials: a tool that touches `os` is refused by
  the review (entry 12).
- The validator's child process gets no secrets (entry 16).
- The Try it tab's server process gets only its declared variables (entry 22).

## 20. Tool names, annotations and the endpoint cap

**Tool names.** They come from the spec's `operationId`, converted in code
(`findPetsByStatus` becomes `find_pets_by_status`), not chosen by the LLM.
- Names stay the same every time a server is regenerated.
- Names never clash between batches.
- The review can match each function to exactly one endpoint by name.

**Annotations.**
- GET tools are marked read-only.
- PUT, PATCH and DELETE are marked destructive.
- POST is marked as neither read-only nor destructive.

The template adds them from the documented method, and the review makes sure
the tool really sends that method. Clients use these marks to ask before running
a tool. Claude Desktop showed an approval prompt for `delete_pet`, and the Try
it tab offers only read-only tools by default.

**Endpoint cap.** At most 40 endpoints by default, and they can be chosen with
`--tags`, `--paths` and `--max-tools`.
- Every tool's description goes into the AI's context.
- With hundreds of tools, it picks the wrong one more often and has less room
  left for the conversation.

## 21. httpx inside generated servers

**Decision.** `_request` uses httpx, with a 30-second timeout.

**Why.**
- httpx sends every request through one function, `Client.send`. The validator
  replaces that function with a recorder (entry 15).
- It supports sync and async, if tools become async later.

**Honest note.** `requests` would also work for the HTTP calls. The deciding
reason is the single function to replace. A generated server needs only two
packages: `mcp` and `httpx`.

## 22. The Try it tab: read-only by default

**Decision.** The Try it tab runs an agent loop:
- Groq gets the generated server's tools over MCP, with `ChatGroq.bind_tools`.
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

## 23. Groq as the LLM

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

## 24. Streamlit for the web UI

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
- Its hosting cannot run containers, which the sandbox in entry 16 would need.

## 25. Pinned versions and CI

**Decision.**
- Every dependency is pinned to an exact version in `requirements.txt`.
- GitHub Actions runs the whole test suite on every push.

**Why.**
- A fresh install (a new laptop, CI, a hosting service) gets the same versions
  the tests passed with.
- This mattered in practice: installing FastAPI upgraded Starlette, which `mcp`
  and `streamlit` also use. Starlette is now pinned on purpose.

**Trade-off.** Upgrades are manual: change a version, reinstall, run the tests.

## 26. Known weaknesses

What is weak today, and why it was left that way:

| Weakness | Why it is acceptable now | Fix |
|---|---|---|
| Review plus isolation is still not a sandbox | The public demo runs only the built-in examples | Container with no network (entry 16) |
| The validator runs whatever it is sent; it relies on its caller having run the review | Only the pipeline calls it, and it listens on 127.0.0.1 by default | The validator renders the trusted header itself and applies the review's blocking rules before running anything |
| Neither agent has authentication | Both listen on 127.0.0.1 by default, so only this machine can reach them | Add auth before exposing them (`..._HOST=0.0.0.0`) |
| Instruction-like text detection is a phrase list | It only warns; a person judges | Better detection; it can never be complete |
| Responses are not checked against the real API | Checking them would mean calling real APIs with real credentials | Optional live check of read-only tools |
| One base URL per server; no OAuth flow; no file uploads | They cover most REST APIs; OAuth works with a token you already have | Extend `_request` |
| ChromaDB is more than this needs | It does no harm and would work with a persistent store | Hybrid BM25 retrieval (entry 5) |
| Only one live run at a time (`RUN_LOCK`) | The RAG index and the demo switch are module-level | Per-run state |
