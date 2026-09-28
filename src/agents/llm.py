"""
One shared LLM instance, plus a retry wrapper for rate limits.

Even generous free tiers (Groq: 30 RPM) can be hit if several analyst
calls fire in the same instant, or if the provider's servers are briefly
overloaded (503). invoke_with_backoff() catches both failure modes and
retries with increasing delay, instead of crashing the whole graph.
"""
import time
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Groq's free tier gives 30 requests/minute (vs Gemini's 5 RPM on newer
# models), plus Groq's LPU hardware returns responses much faster - both
# matter a lot for a graph that fires several LLM calls per run.
analyst_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
manager_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)


def invoke_with_backoff(structured_llm, prompt: str, max_retries: int = 6, base_delay: int = 12):
    """
    Call structured_llm.invoke(prompt), retrying on rate-limit (429) errors
    with exponential backoff: base_delay, base_delay*2, base_delay*3, ...

    Any error that ISN'T a rate limit is re-raised immediately - we only
    want to silently retry the specific failure mode we understand.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return structured_llm.invoke(prompt)
        except Exception as e:
            message = str(e)
            is_retryable = any(
                marker in message
                for marker in ["RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503"]
            )
            if not is_retryable or attempt == max_retries:
                raise

            wait_seconds = base_delay * attempt
            print(
                f"  [rate limit] attempt {attempt}/{max_retries} failed, "
                f"waiting {wait_seconds}s before retrying..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Exceeded max retries due to persistent rate limiting.")
