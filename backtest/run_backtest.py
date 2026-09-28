"""
Backtest loop (Phase F): replays the last N trading days, making a fresh
portfolio decision each day using ONLY price data that would have been
available up to that point (no look-ahead), then measures what actually
happened to that decision the next trading day.

Two honest simplifications versus the live main.py pipeline, both
called out explicitly rather than hidden:

1. NO HISTORICAL NEWS. Our free news tool (yfinance's .news) only
   returns CURRENT headlines regardless of which date we ask about -
   there is no way to fetch news "as it existed" on a past date without
   a paid historical news API. Feeding today's news into a decision
   simulated three weeks ago would be factual data leakage (the exact
   problem the HedgeAgents paper flags in their own methodology). So
   this backtest reasons on price action alone, for every simulated day.

2. NO PER-DAY LLM RATIONALE. To keep the total LLM call count sane on a
   free-tier rate limit, we skip generating prose explanations for each
   of the ~20-30 historical days and keep only the real, math-computed
   weights. main.py's live run still produces full rationales.
"""
import json
import os

import numpy as np
import yfinance as yf

from src.schemas import AnalystVerdict
from src.agents.llm import analyst_llm, invoke_with_backoff
from src.agents.fund_manager import TICKERS, _signal_to_expected_return
from src.tools.price_data import summarize_price_slice, _check_volatility_trigger
from src.tools.portfolio_math import _optimize_core

BACKTEST_DAYS = 20      # simulated trading days to replay
FETCH_PERIOD = "90d"    # buffer so even the earliest simulated day has rolling history
LOG_PATH = "backtest/backtest_log.jsonl"
# Originally 2.0 - backtesting against a naive equal-weight baseline showed
# this made the optimizer nearly ignore analyst signals entirely, since the
# variance penalty (risk_aversion * variance_gap) dwarfed the maximum
# possible expected-return signal (~0.15) for any asset as volatile as
# crypto. Lowered to let genuine conviction actually move allocations.
RISK_AVERSION = 0.3


def fetch_price_history():
    tickers = list(TICKERS.values())
    data = yf.download(tickers, period=FETCH_PERIOD, progress=False)["Close"]
    return data.dropna()


def get_verdict(ticker: str, asset_label: str, price_summary: str) -> AnalystVerdict:
    structured_llm = analyst_llm.with_structured_output(AnalystVerdict)
    prompt = (
        f"You are a {asset_label} market analyst.\n\n"
        f"Price data:\n{price_summary}\n\n"
        "No news data is available for this historical simulation - base "
        "your signal, confidence, and reasoning on price action alone."
    )
    return invoke_with_backoff(structured_llm, prompt)


def run_backtest(prices=None, verdict_fn=get_verdict, backtest_days: int = BACKTEST_DAYS, log_path: str = LOG_PATH):
    """
    prices and verdict_fn are injectable so this function can be unit
    tested with synthetic data and a fake (network-free) verdict source -
    see test_backtest.py. Real runs just call run_backtest() with no args.
    """
    if prices is None:
        prices = fetch_price_history()

    tickers = list(TICKERS.values())
    n = len(prices)

    if n < backtest_days + 10:
        raise RuntimeError(
            f"Not enough price history ({n} rows) for a {backtest_days}-day "
            "backtest with rolling buffer - use a shorter backtest_days or a longer FETCH_PERIOD."
        )

    daily_returns_full = prices[tickers].pct_change().dropna()

    # -1 so the LAST simulated day still has a "next day" to score its decision against
    start_idx = n - backtest_days - 1
    records = []

    for t in range(start_idx, n - 1):
        sim_date = prices.index[t]
        next_date = prices.index[t + 1]

        # --- Step 1: each analyst sees ONLY data up to and including day t ---
        expected_returns = {}
        for label, ticker in TICKERS.items():
            closes_so_far = prices[ticker].iloc[: t + 1]
            summary = summarize_price_slice(closes_so_far, ticker)
            verdict = verdict_fn(ticker, label, summary)
            expected_returns[ticker] = _signal_to_expected_return(verdict.signal, verdict.confidence)

        # --- Step 2: portfolio optimization using ONLY returns up to day t ---
        returns_slice = daily_returns_full.iloc[: t + 1]
        cov_matrix = returns_slice.cov().values * 252
        mu = np.array([expected_returns[tk] for tk in tickers])
        weights_arr = _optimize_core(mu, cov_matrix, risk_aversion=RISK_AVERSION)
        weights = {tk: float(w) for tk, w in zip(tickers, weights_arr)}

        # --- Step 3: risk check using only the last 5 days up to day t ---
        # IMPORTANT: this must be a FIXED-SIZE recent window, matching the
        # live pipeline's period="5d" behavior - NOT the full growing
        # history up to t. Using the full history would mean one big move
        # anywhere in the past stays "detected" forever, since the window
        # never shrinks back down. (This was a real bug caught by testing
        # on synthetic data before running against live prices.)
        risk_triggered = False
        risk_reason = []
        for tk in tickers:
            recent_window = prices[tk].iloc[: t + 1].iloc[-5:]
            result = _check_volatility_trigger(recent_window)
            if result["triggered"]:
                risk_triggered = True
                risk_reason.append(tk)

        if risk_triggered:
            weights = {tk: w * 0.5 for tk, w in weights.items()}
            cash_weight = 1 - sum(weights.values())
        else:
            cash_weight = 0.0

        # --- Step 4: measure what ACTUALLY happened the next day ---
        next_day_asset_returns = daily_returns_full.loc[next_date]
        portfolio_return = sum(weights[tk] * next_day_asset_returns[tk] for tk in tickers)

        record = {
            "date": str(sim_date.date()) if hasattr(sim_date, "date") else str(sim_date),
            "next_date": str(next_date.date()) if hasattr(next_date, "date") else str(next_date),
            "weights": weights,
            "cash_weight": cash_weight,
            "risk_triggered": risk_triggered,
            "risk_reason": risk_reason,
            "portfolio_return": portfolio_return,
        }
        records.append(record)

        print(
            f"{record['date']} -> {record['next_date']} | "
            f"weights={ {k: round(v, 3) for k, v in weights.items()} } "
            f"cash={cash_weight:.2f} risk_triggered={risk_triggered} "
            f"realized_return={portfolio_return:+.4f}"
        )

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    metrics = compute_metrics(records)
    baseline_metrics = compute_baseline_metrics(prices, daily_returns_full, start_idx, n)

    print("\n--- Backtest Summary ---")
    print(f"Days simulated: {metrics['num_days']}")
    print(f"{'Metric':<28}{'Agent System':>15}{'Equal-Weight Baseline':>24}")
    print(f"{'Total return':<28}{metrics['total_return']:>14.2%} {baseline_metrics['total_return']:>23.2%}")
    print(f"{'Annualized return (extrap)':<28}{metrics['annualized_return']:>14.2%} {baseline_metrics['annualized_return']:>23.2%}")
    print(f"{'Sharpe ratio':<28}{metrics['sharpe']:>15.2f}{baseline_metrics['sharpe']:>24.2f}")
    print(f"{'Max drawdown':<28}{metrics['max_drawdown']:>14.2%} {baseline_metrics['max_drawdown']:>23.2%}")
    print(f"\nFull daily log saved to {log_path}")

    return records, metrics, baseline_metrics


def compute_baseline_metrics(prices, daily_returns_full, start_idx: int, n: int) -> dict:
    """
    Naive always-invested, equal-weight-across-all-three-assets baseline,
    with NO analyst signals, NO optimizer, and NO risk veto - a simple
    point of comparison for whether the system's added complexity
    actually earns its keep over the same historical window.
    """
    tickers = list(TICKERS.values())
    records = []
    for t in range(start_idx, n - 1):
        next_date = prices.index[t + 1]
        next_day_returns = daily_returns_full.loc[next_date]
        portfolio_return = sum(next_day_returns[tk] / len(tickers) for tk in tickers)
        records.append({"portfolio_return": portfolio_return})
    return compute_metrics(records)
def compute_metrics(records: list) -> dict:
    """Pure function - takes the records list, returns summary stats. Unit-testable on its own."""
    daily_rets = np.array([r["portfolio_return"] for r in records])
    cumulative = np.cumprod(1 + daily_rets)

    total_return = cumulative[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(daily_rets)) - 1 if len(daily_rets) > 0 else 0.0
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() > 0 else 0.0

    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    return {
        "num_days": len(daily_rets),
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
    }


if __name__ == "__main__":
    run_backtest()