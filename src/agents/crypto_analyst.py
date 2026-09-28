from src.schemas import AnalystVerdict
from src.tools.price_data import get_price_summary
from src.tools.news_data import get_recent_headlines
from src.agents.llm import analyst_llm, invoke_with_backoff

TICKER = "BTC-USD"
structured_llm = analyst_llm.with_structured_output(AnalystVerdict)


def crypto_analyst_node(state: dict) -> dict:
    """
    LangGraph node: fetches real BTC data + news, asks the LLM for a
    structured verdict, and returns it under 'crypto_verdict' so the
    Fund Manager can read it later.
    """
    price_summary = get_price_summary(TICKER)
    headlines = get_recent_headlines(TICKER)

    prompt = (
        f"You are a cryptocurrency market analyst specializing in Bitcoin.\n\n"
        f"Price data:\n{price_summary}\n\n"
        f"{headlines}\n\n"
        f"Based only on this information, give your signal, confidence, "
        f"and reasoning for {TICKER}."
    )

    verdict = invoke_with_backoff(structured_llm, prompt)
    return {"crypto_verdict": verdict}
