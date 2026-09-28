from src.schemas import PortfolioDecision
from src.agents.llm import manager_llm, invoke_with_backoff
from src.tools.portfolio_math import optimize_weights

TICKERS = {"crypto": "BTC-USD", "stock": "AAPL", "forex": "EURUSD=X"}


def _signal_to_expected_return(signal: str, confidence: float) -> float:
    """
    Convert an analyst's qualitative signal into a rough ANNUALIZED
    expected-return estimate the optimizer can use as input.

    Bearish signals are floored at 0 rather than given a negative value:
    since we don't allow short-selling (weights are constrained to [0,1]
    in portfolio_math.py), a negative expected return doesn't let the
    optimizer profit from being bearish - it can only ever choose to hold
    LESS of an asset, never negative amounts. Treating "bearish" as "no
    expected upside" (0) rather than "expect a loss" (negative) is the
    honest way to represent that constraint, but note: this REMOVES a
    penalty that was previously pushing weight away from bearish assets,
    so a bearish asset with low volatility may end up with a similar or
    even larger allocation than before - it's now judged purely on its
    diversification value, with no return expectation either helping or
    hurting it.
    """
    base_by_signal = {"bullish": 0.15, "bearish": -0.15, "neutral": 0.0}
    raw = base_by_signal[signal] * confidence
    return max(raw, 0.0)


def _extract_text(response) -> str:
    """
    Some providers return response.content as a plain string; others
    (we saw this with Gemini) return a list of content blocks. Handle
    both so callers don't have to think about which provider is active.
    """
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def fund_manager_node(state: dict) -> dict:
    crypto = state["crypto_verdict"]
    stock = state["stock_verdict"]
    forex = state["forex_verdict"]

    expected_returns = {
        TICKERS["crypto"]: _signal_to_expected_return(crypto.signal, crypto.confidence),
        TICKERS["stock"]: _signal_to_expected_return(stock.signal, stock.confidence),
        TICKERS["forex"]: _signal_to_expected_return(forex.signal, forex.confidence),
    }
    tickers = list(expected_returns.keys())

    # --- The actual decision: deterministic math, not an LLM guess ---
    weights = optimize_weights(expected_returns, tickers)

    # --- The LLM's only job now: explain the math-driven decision in
    #     plain English, referencing the analysts it was computed from ---
    rationale_prompt = (
        "You are the Fund Manager. A mean-variance portfolio optimizer has "
        "computed this allocation based on each asset's expected return, "
        "historical volatility, and correlation with the others:\n"
        f"- Crypto (BTC-USD): {weights[TICKERS['crypto']]:.1%}\n"
        f"- Stocks (AAPL): {weights[TICKERS['stock']]:.1%}\n"
        f"- Forex (EURUSD=X): {weights[TICKERS['forex']]:.1%}\n\n"
        "IMPORTANT constraints of this system, which your explanation must "
        "respect and never contradict:\n"
        "- Short-selling is NOT allowed. Every weight is a long-only holding "
        "between 0% and 100%. There is no 'short position' in this system.\n"
        "- A bearish analyst signal is treated as a NEUTRAL (zero) expected "
        "return, not a negative one, since a long-only position can't profit "
        "from an asset falling. So a bearish asset can still receive a large "
        "weight purely because it has low volatility or low correlation with "
        "the other holdings (a diversification benefit) - NOT because the "
        "optimizer expects it to rise, and NOT because it is being shorted.\n\n"
        "The analysts' underlying views that fed into this were:\n"
        f"Crypto: {crypto.signal} (confidence {crypto.confidence:.2f}) - {crypto.reasoning}\n"
        f"Stock: {stock.signal} (confidence {stock.confidence:.2f}) - {stock.reasoning}\n"
        f"Forex: {forex.signal} (confidence {forex.confidence:.2f}) - {forex.reasoning}\n\n"
        "In 2-3 sentences, explain this allocation in plain English. If an "
        "asset with a bearish or neutral signal received significant weight, "
        "attribute that to its low volatility/diversification value - never "
        "describe it as shorting or as the optimizer expecting it to rise."
    )
    response = invoke_with_backoff(manager_llm, rationale_prompt)
    rationale = _extract_text(response)

    decision = PortfolioDecision(
        crypto_weight=weights[TICKERS["crypto"]],
        stock_weight=weights[TICKERS["stock"]],
        forex_weight=weights[TICKERS["forex"]],
        rationale=rationale,
    )
    return {"portfolio_decision": decision}