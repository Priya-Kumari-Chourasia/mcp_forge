"""
This file is the ONLY place in the whole project that talks to an LLM.

Every node just calls call_llm(prompt) and gets text back. None of them
need to know whether that text came from Groq or from the offline stub,
so switching backends is one change here, not five across the project.

Two backends:
  - real:  Groq, used when USE_REAL_LLM=true (the default) AND a
           GROQ_API_KEY is available.
  - stub:  canned answers for one tiny API (catfact.ninja). Free, instant,
           offline. Used when USE_REAL_LLM=false or no key is set, so the
           whole graph can be run and debugged with zero setup.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"

USE_REAL_LLM = os.getenv("USE_REAL_LLM", "true").lower() == "true"
if USE_REAL_LLM and not os.getenv("GROQ_API_KEY"):
    print("[llm] No GROQ_API_KEY found - falling back to the offline stub LLM.")
    USE_REAL_LLM = False

_llm = None  # created on first real call, so stub mode needs no Groq setup at all


def call_llm(prompt: str) -> str:
    if USE_REAL_LLM:
        return _real_llm_call(prompt)
    return _fake_llm_call(prompt)


def _strip_code_fences(text: str) -> str:
    """LLMs often wrap answers in ```python ... ``` even when told not to."""
    text = text.strip()
    if text.startswith("```"):
        end_idx = text.rfind("```")
        if end_idx > 3:
            text = text[3:end_idx]
        for lang in ("json", "python", "py"):
            if text.startswith(lang):
                text = text[len(lang):]
                break
    return text.strip()


def _real_llm_call(prompt: str) -> str:
    global _llm
    from groq import RateLimitError
    if _llm is None:
        from langchain_groq import ChatGroq
        # timeout: a request that hangs fails after 60s instead of blocking the run forever.
        _llm = ChatGroq(model=MODEL_NAME, api_key=os.environ["GROQ_API_KEY"], temperature=0,
                          timeout=60, max_retries=2)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = _llm.invoke(prompt)
            return _strip_code_fences(response.content)
        except RateLimitError as e:
            if "tokens per day" in str(e).lower() or "TPD" in str(e):
                raise RuntimeError(
                    "Groq's daily token quota is exhausted for this model. "
                    "This isn't fixable by retrying - wait for the quota to reset, "
                    "or reduce how much you test today."
                ) from e
            wait_seconds = 20 * attempt
            print(f"  [rate limit] short-term limit hit, waiting {wait_seconds}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait_seconds)
    raise RuntimeError("Gave up after repeated rate-limit retries.")


def _fake_llm_call(prompt: str) -> str:
    """Canned answers for the cat-fact API, whatever docs are passed in."""
    if "Analyze these API docs" in prompt:               # orchestrator
        return "simple"
    if "List every API endpoint" in prompt:              # discover_endpoints
        return '[{"method": "GET", "path": "/fact"}]'
    if "Is the context sufficient" in prompt:            # grade_retrieval
        return ('{"sufficient": true, "endpoint": {"method": "GET", "path": "/fact", '
                '"base_url": "https://catfact.ninja", "parameters": [], '
                '"description": "Returns a random cat fact"}}')
    # generate_tool's question
    return (
        "import requests\n\n"
        "def get_fact() -> dict:\n"
        '    """Returns a random cat fact."""\n'
        '    response = requests.get("https://catfact.ninja/fact", timeout=30)\n'
        "    response.raise_for_status()\n"
        "    return response.json()\n"
    )
