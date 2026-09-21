"""
This file is the ONLY place in the whole project that talks to an LLM.

Why isolate it like this? Because later, every node (parse_docs_node,
generate_tool_node, and eventually more) just calls call_llm(prompt) and
gets text back. None of them need to know or care whether that text came
from Groq, Gemini, Claude, or a fake stub. So today you can run the whole
pipeline for free with fake answers, and next week you can make ONE change
here (not five changes scattered across the project) to go live.

Right now call_llm() returns canned, realistic-looking answers instead of
calling a real API. This lets you see the entire graph run end-to-end,
understand exactly what data moves where, and debug your graph logic —
all with zero cost and zero API key setup.
"""

import os
import time
from groq import RateLimitError
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

USE_REAL_LLM = os.getenv("USE_REAL_LLM", "true").lower() == "true"

_llm = None
if USE_REAL_LLM:
    _llm = ChatGroq(
    model="openai/gpt-oss-120b",   # changed from llama-3.3-70b-versatile
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)

def call_llm(prompt: str) -> str:
    if USE_REAL_LLM:
        return _real_llm_call(prompt)
    return _fake_llm_call(prompt)



def _real_llm_call(prompt: str) -> str:
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = _llm.invoke(prompt)
            text = response.content
            if text.startswith("```"):
                end_idx = text.rfind("```")
                if end_idx > 3:
                    text = text[3:end_idx]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            return text
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
    if "Analyze these API docs" in prompt:  # orchestrator's question
        # Simple docs → simple path
        if "weather" in prompt.lower() or "random" in prompt.lower():
            return "simple"
        else:
            return "complex"
    
    # kept as a fallback so you can flip USE_REAL_LLM=false anytime
    # and go back to free, instant, offline testing
    elif "Extract" in prompt:
        return """{"method": "GET", "path": "/fact", "base_url": "https://catfact.ninja", "params": [], "description": "Returns a random cat fact"}"""
    else:
        return 'def get_cat_fact():\n    return {"fact": "Cats sleep 70% of their lives"}\n'


   