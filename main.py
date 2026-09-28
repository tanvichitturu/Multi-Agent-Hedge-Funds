from src.graph import build_graph


def main():
    app = build_graph()
    result = app.invoke(
        {
            "crypto_verdict": None,
            "stock_verdict": None,
            "forex_verdict": None,
            "portfolio_decision": None,
        }
    )

    print("\n--- Analyst Verdicts ---")
    print("Crypto:", result["crypto_verdict"])
    print("Stock: ", result["stock_verdict"])
    print("Forex: ", result["forex_verdict"])
    decision = result["portfolio_decision"]
    assessment = result["risk_assessment"]
    print("\n--- Risk Assessment ---")
    print(f"Extreme Market Conference triggered: {assessment.triggered}")
    print(f"Reason: {assessment.reason}")
    decision = result["portfolio_decision"]
    label = "VETOED (defensive override applied)" if assessment.triggered else "Approved"
    print(f"\n--- Fund Manager Decision [{label}] ---")
    print(f"Crypto: {decision.crypto_weight:.1%}")
    print(f"Stock:  {decision.stock_weight:.1%}")
    print(f"Forex:  {decision.forex_weight:.1%}")
    print(f"Cash:   {decision.cash_weight:.1%}")
    print(f"\nRationale: {decision.rationale}")


if __name__ == "__main__":
    main()
