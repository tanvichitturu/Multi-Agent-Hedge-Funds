from src.schemas import AnalystVerdict
from src.tools.price_data import get_price_summary
from src.tools.news_data import get_recent_headlines
from src.agents.llm import analyst_llm, invoke_with_backoff

TICKER = "EURUSD=X"
structured_llm = analyst_llm.with_structured_output(AnalystVerdict)


def forex_analyst_node(state: dict) -> dict:
    """LangGraph node for the forex analyst (EUR/USD as a stand-in for the paper's YUAN pair)."""
    price_summary = get_price_summary(TICKER)
    headlines = get_recent_headlines(TICKER)

    prompt = (
        f"You are a foreign exchange analyst specializing in major currency pairs.\n\n"
        f"Price data:\n{price_summary}\n\n"
        f"{headlines}\n\n"
        f"Based only on this information, give your signal, confidence, "
        f"and reasoning for {TICKER}."
    )

    verdict = invoke_with_backoff(structured_llm, prompt)
    return {"forex_verdict": verdict}
