"""
SECURITY AGENT - reviews generated tool code BEFORE anything runs it.

A second A2A agent, built exactly like the validator (see a2a_server.py), so
the pipeline discovers it the same way: by its Agent Card.

Why a separate agent, and why it goes first:
  - The validator RUNS the generated code (in a child process that is
    isolation, not a sandbox). Dangerous code has to be stopped before that.
  - This agent never runs code: it reads it with ast (security_rules.py). So it
    can review anything, and a finding never depends on what the code does
    when it runs.
  - Its verdict is reported separately from the validator's: "is it safe to
    run" and "does it work" are different questions with different owners.

There is no LLM in this agent. The docs being processed may contain prompt
injection; fixed rules cannot be talked into approving something.

The message: the generated tool functions only (text part; the server header
is the fixed template and is not reviewed) and the endpoints they implement,
each with its tool "name", "method" and "path" (data part: {"endpoints": [...]}).
The verdict: {"passed": bool, "errors": [...], "warnings": [...]}.

Run it:  python security_agent.py      (listens on http://127.0.0.1:8001)
"""

import os

import a2a_server
import security_rules
from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentInterface, AgentSkill


def build_agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="MCP-Forge Security Reviewer",
        description="Reviews generated MCP tool code without running it: each tool may make only its own "
                    "documented request and use only allowlisted names. Flags instruction-like text in tool "
                    "descriptions (tool poisoning).",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(url=base_url, protocol_binding="JSONRPC", protocol_version="1.0"),
        ],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/x-python", "application/json"],
        default_output_modes=["text/plain", "application/json"],
        skills=[AgentSkill(
            id="review_mcp_tools",
            name="Security review of MCP tools",
            description="Send the generated tool functions (without the server header) as a text part and "
                        "the endpoints as a data part ({\"endpoints\": [{\"name\", \"method\", \"path\", ...}]}). "
                        "The code is never executed. Returns a pass/fail verdict with one structured error per "
                        "problem, plus warnings that do not block.",
            tags=["security", "mcp", "code", "static-analysis"],
        )],
    )


class SecurityReviewExecutor(a2a_server.CheckExecutor):
    """The agent's own logic: the rules in security_rules.py."""
    artifact_prefix = "review"
    log_name = "security"

    def check(self, code, endpoints):
        return security_rules.review(code, endpoints)


def create_app(public_url: str):
    return a2a_server.create_app(build_agent_card(public_url), SecurityReviewExecutor(),
                                 title="MCP-Forge Security Reviewer",
                                 description="An A2A agent that reviews generated MCP tool code without running it.")


def serve_in_thread(host: str = "127.0.0.1", port: int = 0):
    """Starts the reviewer on a background thread; returns (base_url, server)."""
    return a2a_server.serve_in_thread(create_app, host, port)


if __name__ == "__main__":
    # Same settings as the validator's, under SECURITY_AGENT_*.
    port = int(os.getenv("SECURITY_AGENT_PORT", "8001"))
    a2a_server.run_standalone(create_app, "Security agent", os.getenv("SECURITY_AGENT_HOST", "127.0.0.1"), port,
                              os.getenv("SECURITY_AGENT_PUBLIC_URL", f"http://localhost:{port}/"))
