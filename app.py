"""
MCP-Forge web UI.   Run with:   streamlit run app.py

Shows the pipeline working step by step: the agentic RAG rounds, the A2A
call to the validator agent, the self-correcting loop, and the MCP server
that comes out.

Two modes:
  - Recorded run: replays a saved run from demo_runs/. Instant, needs no API
    key, cannot hit a rate limit.
  - Live run: runs the pipeline now, with the real LLM.

Setting PUBLIC_DEMO=true (for a public deployment) limits live runs to the
built-in examples and caps how many can run per day. The validator executes
LLM-generated code, so a public site should not accept arbitrary input.
"""

import datetime
import os
import re
import time

import streamlit as st

st.set_page_config(page_title="MCP-Forge", layout="wide")

# On a hosting service, settings arrive as "secrets". Copy them into the
# environment BEFORE importing the pipeline, because llm.py reads them on import.
for _key in ("GROQ_API_KEY", "USE_REAL_LLM", "EMBEDDINGS", "PUBLIC_DEMO", "MAX_LIVE_RUNS_PER_DAY"):
    try:
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
    except Exception:
        pass        # no secrets file: running locally, .env is used instead

import requests                                              # noqa: E402
import llm                                                   # noqa: E402
import pipeline_runner as runner                             # noqa: E402
from cli import prepare_docs                                 # noqa: E402

PUBLIC_DEMO = os.getenv("PUBLIC_DEMO", "false").lower() == "true"
MAX_LIVE_RUNS_PER_DAY = int(os.getenv("MAX_LIVE_RUNS_PER_DAY", "40"))

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
            st.info("Demo switch is ON: the first generated tool was broken on purpose (its URL lost its host) "
                    "so that the self-correcting loop has an error to fix. The validator's error and the "
                    "repair on the next attempt are real.")
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
                [{"Method": e.get("method"), "Path": e.get("path"), "Description": e.get("description"),
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
            tools = re.findall(r"@server\.tool\(\)\s+(?:async\s+)?def (\w+)", code)
            st.markdown(f"**Generated MCP server - {len(tools)} tool(s):** " + ", ".join(f"`{t}`" for t in tools))
            if finished:
                st.download_button("Download server.py", code, file_name="server.py", key=f"download-{run_id}")
            st.code(code, language="python")
        else:
            st.write("The server is packaged at the end of the run.")

    with slots["log"].container():
        st.code("".join(e["log"] for e in events) or "(no output yet)", language="text")


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
    sources = list(EXAMPLES) if (PUBLIC_DEMO or mode == "Recorded run") else list(EXAMPLES) + ["My own docs"]
    choice = st.selectbox("Documentation", sources)
    if PUBLIC_DEMO:
        pass
    elif mode == "Recorded run":
        st.caption("To test your own URL or pasted docs, switch Mode to **Live run**, "
                   "then pick **My own docs** at the bottom of this list.")
    elif choice != "My own docs":
        st.caption("To test your own URL or pasted docs, pick **My own docs** at the bottom of this list.")

    own_text, own_url = "", ""
    if choice == "My own docs":
        own_url = st.text_input("URL of an OpenAPI spec or docs page")
        own_text = st.text_area("...or paste the documentation", height=180)

    inject_bug = st.toggle("Demo: break the first attempt on purpose",
                           help="Strips the host from one generated URL so the self-correcting loop has an error "
                                "to fix. Without it, the model usually gets the code right first time.")

    save_recording = False
    if mode == "Live run" and not PUBLIC_DEMO and choice != "My own docs":
        save_recording = st.checkbox("Save this run as the recorded demo")

    go = st.button("Run", type="primary", width="stretch")

    st.divider()
    if mode == "Live run":
        st.caption(f"LLM: {'Groq ' + llm.MODEL_NAME if llm.USE_REAL_LLM else 'offline stub (no GROQ_API_KEY set)'}")
    if PUBLIC_DEMO:
        st.caption("Public demo: live runs are limited to the built-in examples, because the validator "
                   "executes generated code.")

slots = {"stages": st.empty(), "progress": st.empty(), "summary": st.empty()}
tabs = st.tabs(["Extraction (agentic RAG)", "Validation (A2A)", "Self-correction", "Generated server", "Log"])
for name, tab in zip(["extraction", "a2a", "loop", "output", "log"], tabs):
    with tab:
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