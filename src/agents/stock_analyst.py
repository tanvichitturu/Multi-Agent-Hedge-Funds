from src.schemas import AnalystVerdict
from src.tools.price_data import get_price_summary
from src.tools.news_data import get_recent_headlines
from src.agents.llm import analyst_llm, invoke_with_backoff

TICKER = "AAPL"
structured_llm = analyst_llm.with_structured_output(AnalystVerdict)


def stock_analyst_node(state: dict) -> dict:
    """LangGraph node for the equities analyst (AAPL as a stand-in for DJIA)."""
    price_summary = get_price_summary(TICKER)
    headlines = get_recent_headlines(TICKER)

    prompt = (
        f"You are an equity market analyst specializing in large-cap US stocks.\n\n"
        f"Price data:\n{price_summary}\n\n"
        f"{headlines}\n\n"
        f"Based only on this information, give your signal, confidence, "
        f"and reasoning for {TICKER}."
    )

    verdict = invoke_with_backoff(structured_llm, prompt)
    return {"stock_verdict": verdict}
