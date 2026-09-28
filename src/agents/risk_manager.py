from src.schemas import RiskAssessment, PortfolioDecision
from src.agents.llm import manager_llm, invoke_with_backoff
from src.agents.fund_manager import TICKERS, _extract_text
from src.tools.price_data import detect_extreme_volatility


def risk_check_node(state: dict) -> dict:
    """
    Runs AFTER the Fund Manager has already decided an allocation. Checks
    each held asset for the paper's Extreme Market Conference trigger
    (>5% single-day move, or >10% cumulative 3-day move). This node only
    WRITES the assessment to state - it doesn't change the allocation
    itself. The conditional edge in graph.py reads this assessment to
    decide whether to route to the veto node or straight to END.
    """
    reasons = []
    any_triggered = False

    for label, ticker in TICKERS.items():
        result = detect_extreme_volatility(ticker)
        if result["triggered"]:
            any_triggered = True
            if result["daily_triggered"]:
                reasons.append(
                    f"{ticker} moved {result['max_daily_move_pct']:.1f}% in a single day"
                )
            if result["three_day_triggered"]:
                reasons.append(
                    f"{ticker} moved {result['three_day_move_pct']:.1f}% over the last 3 days"
                )

    reason_text = "; ".join(reasons) if reasons else "No extreme volatility detected."
    assessment = RiskAssessment(triggered=any_triggered, reason=reason_text)
    return {"risk_assessment": assessment}


def route_after_risk_check(state: dict) -> str:
    """
    Conditional edge function: reads the assessment risk_check_node just
    wrote, returns which node to go to next. Returns a plain string, same
    pattern as Module 1's check_size function - this function never
    touches state, it only decides where to go.
    """
    return "veto" if state["risk_assessment"].triggered else "approve"


def risk_veto_node(state: dict) -> dict:
    """
    Only runs if risk_check_node detected extreme volatility. Overrides
    the Fund Manager's allocation: halves every risky weight and moves the
    freed-up capital into cash, then asks the LLM to explain the override
    using the REAL triggering reason (not a guess) - the same lesson from
    Phase C about not letting an LLM invent a mechanism it wasn't given.
    """
    original = state["portfolio_decision"]
    assessment = state["risk_assessment"]

    halved = PortfolioDecision(
        crypto_weight=original.crypto_weight * 0.5,
        stock_weight=original.stock_weight * 0.5,
        forex_weight=original.forex_weight * 0.5,
        cash_weight=1 - (original.crypto_weight * 0.5 + original.stock_weight * 0.5 + original.forex_weight * 0.5),
        rationale=original.rationale,  # will be overwritten below
    )

    prompt = (
        "You are the Fund Manager's risk oversight function. An Extreme "
        "Market Conference has been triggered because: "
        f"{assessment.reason}\n\n"
        "As a result, the original allocation has been automatically cut "
        "in half across all risky assets, with the freed capital moved to "
        "cash:\n"
        f"- Crypto (BTC-USD): {halved.crypto_weight:.1%}\n"
        f"- Stocks (AAPL): {halved.stock_weight:.1%}\n"
        f"- Forex (EURUSD=X): {halved.forex_weight:.1%}\n"
        f"- Cash: {halved.cash_weight:.1%}\n\n"
        "In 2-3 sentences, explain to the reader why this defensive "
        "override happened, referencing the specific volatility event "
        "above. Do not invent any other reason."
    )
    response = invoke_with_backoff(manager_llm, prompt)
    halved.rationale = _extract_text(response)

    return {"portfolio_decision": halved}