# MCP-Forge

**Point it at API documentation. Get back a working, verified MCP (Model Context Protocol) server.**

Built with LangGraph, the Model Context Protocol (MCP), the Agent2Agent (A2A) protocol, and Agentic RAG. Tested end-to-end against a real, live, 19-endpoint public API it had never seen before.

---

## Problem Statement

The Model Context Protocol (MCP) has become a standard way to expose tools and data sources to AI agents. But adopting it comes with a real cost: **every team that wants to expose an API to an agent has to hand-write the MCP server integration themselves** — reading the documentation, figuring out the auth scheme, writing the tool functions, and testing that they actually work.

This is repetitive, error-prone, and scales badly. A team with 20 internal APIs faces 20 rounds of the same manual work. Documentation is often messy, incomplete, or spread across multiple pages.

**MCP-Forge automates this translation.** Point it at documentation — raw prose or a structured OpenAPI spec — and it produces a working MCP server, extracted, generated, and verified, without a human writing the integration code by hand.

---

## Proof, Not Just a Pitch

```
$ python graph.py --url https://petstore3.swagger.io/api/v3/openapi.json

ORCHESTRATOR -> plan: complex
  ROUTING: complex path
STEP 1 (parse_docs) -> OpenAPI spec detected, 19 endpoint(s), no LLM needed
STEP 1b (grade) -> issues: []
  DECISION: extraction complete, moving to generate_tool
STEP 2 (generate_tool) attempt 1 -> code for 19 endpoint(s)
STEP 3 (test via A2A) -> passed=True, output=19 function(s) defined and valid:
  ['put_pet', 'post_pet', 'get_pet_find_by_status', 'get_pet_find_by_tags',
   'get_pet', 'post_pet_update_form', 'delete_pet', 'post_pet_upload_image',
   'get_store_inventory', 'post_store_order', 'get_store_order', 'delete_store_order',
   'post_user', 'post_user_create_with_list', 'get_user_login', 'get_user_logout',
   'get_user', 'put_user', 'delete_user']
  DECISION: passed, finishing

Saved to: generated/api-v3/mcp_server.py | Status: PASSED
```

That's the public Swagger Petstore API — documentation MCP-Forge had never seen, hand-tuned for nothing — passing cleanly on the **first attempt**, with sensible function names inferred from REST conventions it was never explicitly told (`get_pet_find_by_status` from `GET /pet/findByStatus`).

---

## What Makes This Different From "Just Ask an LLM"

A single prompt to an LLM can produce plausible-looking integration code. The problem is **plausible isn't verified**. If the LLM misreads a parameter or misses something in the docs, you don't find out until someone tries to use the generated server and it silently fails.

MCP-Forge closes that loop with two properties a single prompt doesn't have:

1. **Agentic RAG** — the documentation-reading step doesn't trust its own first answer. It checks whether the extraction is actually complete, and re-queries if it isn't.
2. **A2A-verified generation** — the generated code isn't just written, it's checked by a separate agent (communicating over the Agent2Agent protocol) that reports back real pass/fail results, triggering another generation attempt on failure.

---

## Key Features

- **Orchestrator Agent** — reads the docs first and decides whether the API is "simple" or "complex," routing to the appropriate path
- **Deterministic OpenAPI Parsing** — a structured OpenAPI/Swagger spec is parsed directly, with zero LLM calls; extraction only falls back to agentic RAG for unstructured prose documentation
- **Agentic RAG Extraction Loop** — for prose docs, extracts structured endpoint info, grades its own completeness, and automatically retries with a refined prompt if fields are missing
- **Real A2A Communication** — the code-generation agent and the validation agent are **separate, independently running services** that exchange standardized task/message/artifact JSON over HTTP
- **Static Validation, Not Live Execution** — generated code is checked for syntax validity, defined functions, and callability via static analysis (`ast.parse`) rather than actually calling functions that would hit a real, external API. This is a deliberate design choice, not a shortcut (see below).
- **Multi-Endpoint Support** — handles APIs with a single endpoint or dozens, generating one MCP tool function per endpoint
- **Command-Line Interface** — point it at your own docs via `--file`, `--url`, or `--text`; no code editing required
- **Real File Output** — generated servers are saved to `generated/<api-name>/mcp_server.py`, a real deliverable, not console text
- **Swappable LLM Backend** — free/offline stub for development, real Groq (`openai/gpt-oss-20b`) integration for production, one environment variable apart
- **Built to Deploy Free** — designed to run on Oracle Cloud's Always Free tier at zero hosting cost

---

## Why Static Validation, Not Live Execution

Early versions of MCP-Forge actually called each generated function to test it. This broke against a real 19-endpoint API: **every MCP tool this project generates makes a real network call by design** — that's the whole point of an MCP tool. Live-testing that:

- Has **unbounded time** — a slow or hanging real endpoint makes test time undefined, no matter what timeout is set
- Can have **real side effects** — a generated `delete_pet()` function, if actually called during testing, issues a real `DELETE` against a live server
- Is **unreproducible** — a test that depends on a third-party server being up, fast, and unauthenticated isn't a test you can trust

Static validation (syntax check, function existence, callability) catches every real bug class this project has actually hit — syntax errors, missing functions, undefined variables — instantly and deterministically, regardless of API size.

---

## Architecture

```
                        +------------------+
   API docs/spec  --->  |   ORCHESTRATOR   |  (reads docs, decides: simple or complex?)
                        +--------+---------+
                                 |
                 +---------------+----------------+
                 v                                v
          [ simple path ]                  [ complex path ]
                 |                                |
                 +---------------+----------------+
                                 v
                        +------------------+
                        |   PARSE DOCS     |<--+  structured spec: parsed directly
                        |  (OpenAPI or     |   |  prose docs: agentic RAG
                        |   agentic RAG)   |   |  (extract, grade, retry)
                        +--------+---------+   |
                                 v              |
                        +------------------+    |
                        | GRADE EXTRACTION |----+
                        +--------+---------+
                                 | (complete)
                                 v
                        +------------------+
                        |  GENERATE TOOL   |<--+  (writes Python
                        |   (LLM codegen)  |   |   MCP tool functions)
                        +--------+---------+   |
                                 |  HTTP/A2A    |  retry loop:
                                 v              |  regenerate on
                        +------------------+    |  validation failure
                        |  VALIDATOR       |----+
                        | (separate agent, |
                        |  static checks)  |
                        +--------+---------+
                                 | (passed)
                                 v
                        +------------------+
                        |   SAVE OUTPUT    |  writes a real file to disk
                        +------------------+
```

| Piece | What it checks | What happens if it fails |
|---|---|---|
| Grade Extraction | "Did I actually find every required field?" | Re-queries the LLM with a refined prompt |
| Validator (A2A) | "Is the generated code syntactically valid, with real callable functions?" | Sends the real error back, triggers a fresh code-gen attempt |

---

## Tech Stack

- **LangGraph** — orchestration layer; models the pipeline as a state graph with conditional, looping edges
- **Model Context Protocol (MCP)** — the target output format
- **Agent2Agent (A2A) Protocol** — codegen and validation agents communicate as independent services over HTTP using an A2A-style task/message/artifact schema
- **Agentic RAG** — the prose-documentation extraction path retrieves, self-grades, and retries
- **Groq (`openai/gpt-oss-20b`)** — real LLM backend, free-tier, fast inference
- **Flask** — powers the standalone validator A2A service
- **Python 3.13**, `ast` module for static code analysis

---

## How It Works — Stage by Stage

### Stage 1 — Core Pipeline
A minimal two-node LangGraph pipeline: `parse_docs -> generate_tool`, proving the basic flow with a free offline stub LLM.

### Stage 2 — Agentic RAG Retry Loop
Adds `grade_extraction`: checks whether parsed info is complete, loops back to `parse_docs` if not, up to a retry cap.

### Stage 3 — A2A-Verified Code Generation
Introduces a **separate Flask server** (`test_runner_service.py`) exposed over HTTP with an A2A-style message format. Failures trigger automatic regeneration.

### Stage 4 — Orchestrator Agent & Multi-Endpoint Support
Adds an `orchestrator` node that classifies the API as simple or complex before extraction, and hardens the pipeline to handle many endpoints per API, not just one.

### Production Hardening — CLI, Deterministic Parsing, Static Validation
- Added a real CLI (`--file`, `--url`, `--text`) so anyone can point the tool at their own documentation
- Added deterministic OpenAPI spec parsing, bypassing the LLM entirely for structured specs
- Replaced live-execution testing with static validation, after discovering live execution doesn't scale safely to real multi-endpoint APIs (see "Why Static Validation" above)
- Added real file output and connection-error handling with actionable messages

### Stages 5 & 6 (Planned)
- **Stage 5, Packaging**: wrap generated functions as a proper MCP server object with a manifest and one-click install
- **Stage 6, Dashboard UI**: a web interface to submit docs and watch the pipeline run in real time

---

## Project Structure

```
mcp-forge/
|-- state.py                  # Shared state schema
|-- llm.py                    # LLM abstraction - free stub or real Groq call
|-- spec_parser.py            # Deterministic OpenAPI/Swagger spec detection + parsing
|-- cli.py                    # Command-line input: --file, --url, or --text
|-- nodes.py                  # Pipeline nodes: orchestrator, parse, grade, generate, validate, save
|-- graph.py                  # LangGraph wiring - nodes, edges, conditional routing
|-- test_runner_service.py    # Standalone A2A validator (separate process, own HTTP server)
|-- requirements.txt
|-- .env.example
`-- README.md
```

---

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run free, offline (no API key needed)
```bash
python graph.py --text "GET /ping - health check endpoint"
```

### 3. Run with a real LLM and your own docs
1. Get a free API key at console.groq.com (no card required)
2. Copy `.env.example` to `.env`, add `GROQ_API_KEY` and set `USE_REAL_LLM=true`
3. Start the validator, in its own terminal (restart it after any code change):
   ```bash
   python test_runner_service.py
   ```
4. Point it at your own documentation:
   ```bash
   python graph.py --file my_api_docs.txt
   python graph.py --url https://your-api.com/openapi.json
   python graph.py --text "GET /endpoint - description"
   ```

---


## Author

**Priya Kumari** - M.Tech, Computer Science and Engineering, IIT Jodhpur
Researching Extreme Multi-Label Classification (XMC) under Dr. Yashaswi Verma

Built as a hands-on exploration of agentic AI systems, with every stage understood and debugged from first principles rather than scaffolded from a template.
