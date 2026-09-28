"""
Standalone test: forces a fake 'crisis' scenario to prove the veto path
works end-to-end, including a REAL LLM call - without needing to wait
for an actual market crash or touch main.py's real pipeline.
"""
from src.schemas import PortfolioDecision, RiskAssessment
from src.agents.risk_manager import risk_veto_node

fake_state = {
    "portfolio_decision": PortfolioDecision(
        crypto_weight=0.60,
        stock_weight=0.30,
        forex_weight=0.10,
        cash_weight=0.0,
        rationale="Original bullish allocation before the crash (this is fake test data).",
    ),
    "risk_assessment": RiskAssessment(
        triggered=True,
        reason="BTC-USD moved -18.4% in a single day (this is a fake test scenario)",
    ),
}

result = risk_veto_node(fake_state)
decision = result["portfolio_decision"]

print("--- Forced Veto Test ---")
print(f"Crypto: {decision.crypto_weight:.1%}  (should be half of 60% = 30%)")
print(f"Stock:  {decision.stock_weight:.1%}  (should be half of 30% = 15%)")
print(f"Forex:  {decision.forex_weight:.1%}  (should be half of 10% = 5%)")
print(f"Cash:   {decision.cash_weight:.1%}  (should be the freed 50%)")
print(f"\nReal LLM rationale: {decision.rationale}")