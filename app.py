"""
MCP-Forge web UI.   Run with:   streamlit run app.py

Shows the pipeline working step by step: the agentic RAG rounds, the A2A
call to the validator agent, the self-correcting loop, and the MCP server
that comes out.

Two modes:
  - Recorded run: replays a saved run from demo_runs/. Instant, needs no API
    key, cannot hit a rate limit.
  - Live run: runs the pipeline now, with the real LLM.

Setting PUBLIC_DEMO=true (for a public deployment) caps how many live runs can
happen per day and limits visitors to the built-in examples. ALLOW_CUSTOM_DOCS=true
lets them paste their own docs as well - only turn that on once generated code
runs in an isolated container. The validator's child process blocks the usual
ways to reach the network or start programs and hides secret environment
variables, but determined code can get around those blocks and can read files
(such as the file holding the Groq key), so it is not a security sandbox.
"""

import datetime
import hashlib
import json
import os
import re
import tempfile
import time

import streamlit as st

st.set_page_config(page_title="MCP-Forge", layout="wide")

# On a hosting service, settings arrive as "secrets". Copy them into the
# environment BEFORE importing the pipeline, because llm.py reads them on import.
for _key in ("GROQ_API_KEY", "USE_REAL_LLM", "EMBEDDINGS", "PUBLIC_DEMO", "MAX_LIVE_RUNS_PER_DAY",
             "ALLOW_CUSTOM_DOCS"):
    try:
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
    except Exception:
        pass        # no secrets file: running locally, .env is used instead

import requests                                              # noqa: E402
import llm                                                   # noqa: E402
import pipeline_runner as runner                             # noqa: E402
import server_template                                       # noqa: E402
import try_it                                                # noqa: E402
from cli import prepare_docs                                 # noqa: E402

PUBLIC_DEMO = os.getenv("PUBLIC_DEMO", "false").lower() == "true"
# Off by default: on a public demo, docs pasted by a stranger could steer the LLM
# into writing code that leaks secrets (see the note at the top of this file).
# Running locally (PUBLIC_DEMO=false), your own docs are always allowed.
ALLOW_CUSTOM_DOCS = os.getenv("ALLOW_CUSTOM_DOCS", "false").lower() == "true"
MAX_LIVE_RUNS_PER_DAY = int(os.getenv("MAX_LIVE_RUNS_PER_DAY", "40"))
# Each question in the "Try it" tab costs LLM calls, so a public demo caps them per day.
MAX_CHAT_MESSAGES_PER_DAY = int(os.getenv("MAX_CHAT_MESSAGES_PER_DAY", "200"))

EXAMPLES = {
    "Bookstore guide - prose docs (shows agentic RAG)": {
        "slug": "bookstore", "kind": "file", "value": "examples/bookstore_api_long.md"},
    "Swagger Petstore - OpenAPI spec, 19 endpoints": {
        "slug": "petstore", "kind": "url", "value": "https://petstore3.swagger.io/api/v3/openapi.json"},
    "Cat fact - one endpoint": {
        "slug": "catfact", "kind": "text", "value": "GET https://catfact.ninja/fact - returns a random cat fact"},
}

STAGES = ["Orchestrator", "Extract endpoints", "Generate tools", "Validate over A2A", "Package MCP server"]
NODE_TO_STAGE = {
    "orchestrator": 0,
    "parse_docs": 1, "index_docs": 1, "discover_endpoints": 1, "retrieve": 1,
    "grade_retrieval": 1, "rewrite_query": 1, "grade_extraction": 1,
    "generate_tool": 2, "test_via_a2a": 3, "package_server": 4,
}


@st.cache_resource
def live_run_counter():
    return {"day": None, "count": 0}


@st.cache_resource
def chat_counter():
    return {"day": None, "count": 0}


def under_daily_limit(counter, limit: int) -> bool:
    """Counts one use for today and says whether the limit still allows it."""
    today = datetime.date.today().isoformat()
    if counter["day"] != today:
        counter.update(day=today, count=0)
    if counter["count"] >= limit:
        return False
    counter["count"] += 1
    return True


def load_input(kind: str, value: str):
    """Returns (docs_text, source_url)."""
    if kind == "file":
        with open(value, encoding="utf-8") as f:
            return f.read(), None
    if kind == "url":
        response = requests.get(value, timeout=15)
        response.raise_for_status()
        return prepare_docs(response.text), value
    return value, None


# ----------------------------------------------------------------------
# Drawing a run. `events` is everything received so far; the same function
# draws a live run in progress, a finished run, and a replayed recording.
# ----------------------------------------------------------------------

def merged_state(events):
    state = {}
    for event in events:
        state.update({k: v for k, v in event["state"].items() if v is not None})
    return state


def draw(events, slots, finished: bool, run_id: str):
    state = merged_state(events)
    last_node = events[-1]["node"]
    current = NODE_TO_STAGE.get(last_node, -1)
    stopped_early = finished and last_node != "package_server"
    rounds = state.get("rag_trace") or []
    attempts = [a for a in (state.get("attempt_history") or []) if "passed" in a]

    # --- stage tracker ---
    with slots["stages"].container():
        for column, (i, name) in zip(st.columns(len(STAGES)), enumerate(STAGES)):
            if i < current or (finished and i <= current):
                status = ":green[done]"
            elif i == current:
                status = ":orange[running]"
            else:
                status = ":gray[waiting]"
            column.markdown(f"**{i + 1}. {name}**  \n{status}")

    # --- headline numbers ---
    with slots["summary"].container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Input", state.get("input_kind", "-"))
        c2.metric("Plan", state.get("orchestration_plan") or "-",
                  help="The orchestrator's decision. It sets the retry budget for code generation.")
        c3.metric("Endpoints", len(state.get("endpoints") or []))
        c4.metric("Queries rewritten", sum(1 for r in rounds if not r["sufficient"]),
                  help="Retrieval rounds the grader judged insufficient.")
        if finished and not stopped_early:
            c5.metric("Result", "Passed" if state.get("test_passed") else "Failed",
                      f"{len(attempts)} attempt(s)", delta_color="off")
        else:
            c5.metric("Result", "stopped" if stopped_early else "running")
        if state.get("bug_injected"):
            st.info("Demo switch is ON: the first generated tool was broken on purpose (its HTTP method was "
                    "swapped, for example GET to POST) so that the self-correcting loop has an error to fix. "
                    "The validator's error and the repair on the next attempt are real.")
        if state.get("selection_note"):
            st.warning(f"Endpoint selection: {state['selection_note']}.")
        if stopped_early:
            st.warning("The pipeline stopped before generating code: no usable endpoints were extracted.")

    # --- extraction ---
    with slots["extraction"].container():
        if state.get("input_kind") == "OpenAPI spec":
            st.markdown("This input is a structured **OpenAPI spec**, so it is parsed directly in code "
                        "(`spec_parser.py`) with no LLM calls and no retrieval. Agentic RAG is used for prose docs.")
        elif rounds or state.get("rag_candidates"):
            st.markdown("**Agentic RAG.** The docs are chunked and embedded into ChromaDB. For each endpoint the "
                        "pipeline retrieves context, the LLM grades whether it is sufficient, and if not the query "
                        "is rewritten and retrieval runs again.")
            st.dataframe(
                [{"Endpoint": r["endpoint"], "Round": r["round"], "Query": r["query"],
                  "Grader's verdict": "sufficient" if r["sufficient"] else "insufficient",
                  "What was missing": r.get("missing") or ""} for r in rounds],
                width="stretch", hide_index=True)
        endpoints = state.get("endpoints") or []
        if endpoints:
            st.markdown(f"**{len(endpoints)} endpoint(s) extracted**")
            st.dataframe(
                [{"Tool": e.get("name") or "", "Method": e.get("method"), "Path": e.get("path"),
                  "Description": e.get("description"),
                  "Parameters": ", ".join(p.get("name", "") for p in e.get("parameters") or []),
                  "Body fields": ", ".join((e.get("request_body") or {}).get("fields") or []),
                  "Base URL": e.get("base_url")} for e in endpoints],
                width="stretch", hide_index=True)

    # --- validation over A2A ---
    with slots["a2a"].container():
        left, right = st.columns(2)
        with left:
            st.markdown("**1. Discovery - the validator's Agent Card**")
            st.caption("Fetched from `/.well-known/agent-card.json`. The pipeline knows only the agent's address; "
                       "the card tells it what the agent can do and how to call it.")
            st.json(state.get("agent_card") or {}, expanded=2)
        with right:
            st.markdown("**2. SendMessage - tasks sent to the agent**")
            st.caption("Each message carries the generated code (text part) and the endpoints it must implement "
                       "(data part). The reply is a Task; the verdict is in its artifacts.")
            log = "".join(e["log"] for e in events)
            tasks = re.findall(r"A2A: task (\S+) -> (\S+)", log)
            if tasks:
                st.dataframe(
                    [{"Attempt": i + 1, "Task id": task_id[:8] + "...",
                      "Task state": task_state.replace("TASK_STATE_", ""),
                      "Verdict": ("passed" if attempts[i]["passed"] else "failed") if i < len(attempts) else ""}
                     for i, (task_id, task_state) in enumerate(tasks)],
                    width="stretch", hide_index=True)
                st.caption("A task is COMPLETED whenever validation ran, even when the code failed it: "
                           "\"the code is wrong\" is the agent's answer, not a failure of the agent.")
            else:
                st.write("No task sent yet.")

    # --- self-correction ---
    with slots["loop"].container():
        st.markdown("**Self-correcting loop.** When validation fails, the validator's errors and the previous code "
                    "go back into the prompt, and only the batches that failed are regenerated.")
        if not attempts:
            st.write("No validation result yet.")
        for a in attempts:
            title = f"Attempt {a['attempt']} - generated batch(es) {a['batches']} - "
            if a["passed"]:
                st.success(title + "passed")
            else:
                st.error(title + f"failed with {len(a['errors'])} error(s), fed back to the generator:")
                for error in a["errors"]:
                    st.markdown(f"- `{error}`")
        if finished and len(attempts) == 1 and attempts[0]["passed"]:
            st.caption("The code passed on the first attempt, so there was nothing to correct. "
                       "Turn on the demo switch in the sidebar to see the loop repair a broken tool.")

    # --- output ---
    with slots["output"].container():
        code = state.get("server_code")
        if code:
            tools = server_template.tool_names_in(code)      # the public functions = the tools
            st.markdown(f"**Generated MCP server - {len(tools)} tool(s):** " + ", ".join(f"`{t}`" for t in tools))
            check = state.get("mcp_client_check")
            if check and check.get("passed"):
                st.success(f"MCP client check passed: server.py was started over stdio, as Claude Desktop does, "
                           f"and listed {check['tools_listed']} tool(s).")
            elif check:
                st.error(f"MCP client check failed: {check.get('error') or 'missing ' + str(check.get('missing'))}")
            env_names = server_template.auth_env_names(state.get("api_meta") or {})
            if env_names:
                st.caption("Credentials are read from environment variables, never from the AI: "
                           + ", ".join(f"`{n}`" for n in env_names)
                           + ". The generated claude_desktop_config.json has a place for each.")
            if finished:
                st.download_button("Download server.py", code, file_name="server.py", key=f"download-{run_id}")
            st.code(code, language="python")
        else:
            st.write("The server is packaged at the end of the run.")

    with slots["log"].container():
        st.code("".join(e["log"] for e in events) or "(no output yet)", language="text")


# ----------------------------------------------------------------------
# "Try it": chat with the generated server, the way Claude Desktop would
# ----------------------------------------------------------------------

def server_file_for(code: str) -> str:
    """Writes the run's server code to a temporary file the chat can start. Taking
    the code from the run (not from generated/) means a recorded run works too,
    on a machine where that server was never generated."""
    folder = os.path.join(tempfile.gettempdir(), "mcp_forge_try_it")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, hashlib.sha256(code.encode()).hexdigest()[:16] + ".py")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    return path


def render_try_it(events, run_id):
    st.markdown("**Try the generated server.** Ask a question in plain English. An LLM answers it by calling "
                "the server's tools over MCP, the same way Claude Desktop does, and the real API responds.")
    state = merged_state(events) if events else {}
    code = state.get("server_code")
    if not code:
        st.info("Run the pipeline (or replay a recorded run) first: the generated server appears here.")
        return
    model = llm.chat_model()
    if model is None:
        st.info("The chat needs a real LLM to call the tools: set GROQ_API_KEY in .env (or in the app's secrets).")
        return

    meta = state.get("api_meta") or {}
    chats = st.session_state.setdefault("chats", {})
    chat = chats.setdefault(run_id, {"messages": [try_it.system_message(server_template.display_name(meta))],
                                     "display": []})

    changing = sum(1 for e in state.get("endpoints") or [] if (e.get("method") or "").upper() != "GET")
    allow_changes = st.toggle(
        "Allow tools that change data (POST, PUT, PATCH, DELETE)", value=False, disabled=PUBLIC_DEMO,
        key=f"allow-{run_id}",
        help="Off: only read-only tools are offered, so a question cannot change or delete data on the API. "
             "Always off on the public demo.")
    if changing and not allow_changes:
        st.caption(f"{changing} tool(s) that change data are turned off.")

    for entry in chat["display"]:
        with st.chat_message(entry["role"]):
            for step in entry.get("trace", []):
                outcome = "returned an error" if step["error"] else "returned data"
                st.caption(f"Called `{step['tool']}` with `{json.dumps(step['args'])}` - {outcome}")
            st.markdown(entry["content"])

    question = st.chat_input("Ask the API something, e.g. \"Find available pets\"", key=f"chat-{run_id}")
    if not question:
        return
    if PUBLIC_DEMO and not under_daily_limit(chat_counter(), MAX_CHAT_MESSAGES_PER_DAY):
        st.error("Today's question limit for this public demo has been reached.")
        return

    chat["display"].append({"role": "user", "content": question})
    env_names = server_template.auth_env_names(meta) + [f"{server_template.env_prefix(meta)}_BASE_URL"]
    try:
        with st.spinner("Calling the server's tools..."):
            outcome = try_it.ask(server_file_for(code), chat["messages"], question, model, allow_changes, env_names)
        chat["display"].append({"role": "assistant", "content": outcome["answer"] or "(no answer)",
                                "trace": outcome["trace"]})
    except Exception as error:
        # Start the LLM's side of the conversation afresh: a half-finished tool call
        # left in it would make the next request invalid.
        chat["messages"] = chat["messages"][:1]
        chat["display"].append({"role": "assistant",
                                "content": f"The chat stopped with an error: {try_it.describe_error(error)}"})
    st.rerun()


STUCK_AFTER_SECONDS = 120


def show_progress(slot, progress):
    """The 'is it alive?' box, refreshed about once a second during a live run."""
    idle, elapsed = progress["idle_seconds"], progress["elapsed_seconds"]
    with slot.container():
        if idle >= STUCK_AFTER_SECONDS:
            st.error(f"No output for {idle} seconds. This run is probably stuck - press Stop (top right) "
                     f"and try a smaller input or a recorded run.")
        elif "[rate limit]" in (progress["tail"].splitlines() or [""])[-1]:
            st.warning(f"Waiting for the LLM's rate limit to reset. Running for {elapsed}s; "
                       f"last output {idle}s ago.")
        else:
            st.info(f"Working. Running for {elapsed}s; last output {idle}s ago.")
        if progress["tail"]:
            st.code(progress["tail"], language="text")


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------

st.title("MCP-Forge")
st.markdown("Turns API documentation into a validated **MCP server**. A LangGraph pipeline extracts the endpoints "
            "(with **agentic RAG** for prose docs), generates one tool per endpoint, and sends the code to a "
            "separate validator agent over the **A2A** protocol. Validation errors are fed back until the code passes.")

with st.sidebar:
    st.header("Run")
    mode = st.radio("Mode", ["Recorded run", "Live run"],
                    help="A recorded run replays a saved run instantly. A live run calls the LLM now.")
    custom_docs_enabled = ALLOW_CUSTOM_DOCS or not PUBLIC_DEMO
    sources = list(EXAMPLES) + (["My own docs"] if mode == "Live run" and custom_docs_enabled else [])
    choice = st.selectbox("Documentation", sources)
    if custom_docs_enabled and mode == "Recorded run":
        st.caption("To test your own URL or pasted docs, switch Mode to **Live run**, "
                   "then pick **My own docs** at the bottom of this list.")
    elif custom_docs_enabled and choice != "My own docs":
        st.caption("To test your own URL or pasted docs, pick **My own docs** at the bottom of this list.")

    own_text, own_url = "", ""
    if choice == "My own docs":
        own_url = st.text_input("URL of an OpenAPI spec or docs page")
        own_text = st.text_area("...or paste the documentation", height=180)

    inject_bug = st.toggle("Demo: break the first attempt on purpose",
                           help="Swaps the HTTP method of one generated tool so the self-correcting loop has an "
                                "error to fix. Without it, the model usually gets the code right first time.")

    save_recording = False
    if mode == "Live run" and not PUBLIC_DEMO and choice != "My own docs":
        save_recording = st.checkbox("Save this run as the recorded demo")

    go = st.button("Run", type="primary", width="stretch")

    st.divider()
    if mode == "Live run":
        st.caption(f"LLM: {'Groq ' + llm.MODEL_NAME if llm.USE_REAL_LLM else 'offline stub (no GROQ_API_KEY set)'}")
    if PUBLIC_DEMO:
        st.caption(f"Public demo: up to {MAX_LIVE_RUNS_PER_DAY} live runs per day across all visitors. "
                   "Very long docs are cut to their first 60,000 characters.")

slots = {"stages": st.empty(), "progress": st.empty(), "summary": st.empty()}
tabs = dict(zip(["extraction", "a2a", "loop", "output", "try_it", "log"],
                st.tabs(["Extraction (agentic RAG)", "Validation (A2A)", "Self-correction", "Generated server",
                         "Try it", "Log"])))
for name in ("extraction", "a2a", "loop", "output", "log"):
    with tabs[name]:
        slots[name] = st.empty()

if go:
    st.session_state.pop("events", None)
    run_id = str(time.time())
    events = []

    if mode == "Recorded run":
        recorded = runner.load_recording(EXAMPLES[choice]["slug"], inject_bug)
        if recorded is None:
            st.error("There is no recording for this choice yet. Run it once in Live mode on your own machine "
                     "with \"Save this run as the recorded demo\" ticked, then commit the file in demo_runs/.")
        else:
            for event in recorded:
                events.append(event)
                draw(events, slots, finished=False, run_id=run_id)
                time.sleep(0.35)
    else:
        counter = live_run_counter()
        today = datetime.date.today().isoformat()
        if counter["day"] != today:
            counter.update(day=today, count=0)

        if PUBLIC_DEMO and counter["count"] >= MAX_LIVE_RUNS_PER_DAY:
            st.error("Today's live-run limit for this public demo has been reached. Use a recorded run instead.")
        elif choice == "My own docs" and not (own_text.strip() or own_url.strip()):
            st.error("Paste some documentation or give a URL first.")
        elif not runner.RUN_LOCK.acquire(blocking=False):
            st.error("Another live run is in progress. Try again in a minute, or use a recorded run.")
        else:
            try:
                counter["count"] += 1
                if choice == "My own docs":
                    docs, source_url = load_input("url", own_url.strip()) if own_url.strip() else (own_text, None)
                else:
                    docs, source_url = load_input(EXAMPLES[choice]["kind"], EXAMPLES[choice]["value"])
                for event in runner.stream_run(docs, source_url, inject_bug):
                    if event["node"] == "progress":
                        show_progress(slots["progress"], event["progress"])
                        continue
                    events.append(event)
                    draw(events, slots, finished=False, run_id=run_id)
                slots["progress"].empty()
                if save_recording and events:
                    path = runner.save_recording(EXAMPLES[choice]["slug"], inject_bug, events)
                    st.sidebar.success(f"Saved {path}")
            except Exception as error:
                slots["progress"].empty()
                events = []          # a crashed run is not drawn as a finished one
                st.error(f"The run stopped with an error: {type(error).__name__}: {error}")
            finally:
                runner.RUN_LOCK.release()

    if events:
        st.session_state["events"] = events
        st.session_state["run_id"] = run_id

if st.session_state.get("events"):
    draw(st.session_state["events"], slots, finished=True, run_id=st.session_state["run_id"])
else:
    slots["summary"].info("Choose a documentation source in the sidebar and press **Run**.")

# Drawn once per page load, after the run, because it holds an input box.
with tabs["try_it"]:
    render_try_it(st.session_state.get("events"), st.session_state.get("run_id"))
