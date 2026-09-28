"""
Logging node: the last stop before END on every path through the graph.
Writes one JSON line per run to logs/decisions.jsonl, capturing every
analyst's reasoning, the risk assessment, and the final decision - the
paper trail your README/demo can point back to as proof of "explainability".

JSONL (one JSON object per line) is used instead of one big JSON file
because it's append-only: each run adds exactly one line, so you never
need to re-read and re-write the whole file's history just to log a
new run. It's also what Phase G's backtest loop will read back later
to compute performance metrics across many runs.
"""
import json
import os
from datetime import datetime, timezone

LOG_PATH = "logs/decisions.jsonl"


def logger_node(state: dict) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analyst_verdicts": {
            "crypto": state["crypto_verdict"].model_dump(),
            "stock": state["stock_verdict"].model_dump(),
            "forex": state["forex_verdict"].model_dump(),
        },
        "risk_assessment": state["risk_assessment"].model_dump(),
        "portfolio_decision": state["portfolio_decision"].model_dump(),
    }

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\n[logged this run to {LOG_PATH}]")

    # This node doesn't change any decision-relevant state - it's a
    # side-effect-only step, so it returns an empty update.
    return {}