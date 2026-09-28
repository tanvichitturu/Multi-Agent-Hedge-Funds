"""
Real portfolio optimization (mean-variance), replacing the LLM's guessed
allocation from Phase B. This is deterministic and auditable - the same
inputs always produce the same weights - which is the paper's Budget
Allocation Conference in spirit, simplified from their full CVaR-based
formulation (Eq. 3-5) to plain mean-variance for tractability.
"""
import numpy as np
import yfinance as yf
from scipy.optimize import minimize


def get_returns_matrix(tickers: list, period: str = "60d"):
    """Fetch daily % returns for each ticker, aligned by date."""
    data = yf.download(tickers, period=period, progress=False)["Close"]
    returns = data.pct_change().dropna()
    return returns


def _optimize_core(mu: np.ndarray, cov_matrix: np.ndarray, risk_aversion: float) -> np.ndarray:
    """
    Pure math, no data fetching - kept separate from optimize_weights() so
    it can be unit tested with synthetic numbers instead of needing a live
    network call every time you want to check the math is right.

    Maximizes: expected_return - risk_aversion * variance
    Subject to: weights sum to 1, each weight in [0, 1] (no shorting)
    """
    n = len(mu)

    def negative_utility(w):
        port_return = w @ mu
        port_variance = w @ cov_matrix @ w
        return -(port_return - risk_aversion * port_variance)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    initial_guess = np.array([1 / n] * n)

    result = minimize(
        negative_utility,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def optimize_weights(expected_returns: dict, tickers: list, risk_aversion: float = 2.0) -> dict:
    """
    expected_returns: {ticker: annualized expected return estimate}
    tickers: list of tickers, in the same order you want weights back in
    risk_aversion: higher = more conservative (penalizes variance more)

    Returns: {ticker: weight}, weights summing to 1.0
    """
    returns_df = get_returns_matrix(tickers)
    cov_matrix = returns_df.cov().values * 252  # annualize daily covariance

    mu = np.array([expected_returns[t] for t in tickers])
    weights = _optimize_core(mu, cov_matrix, risk_aversion)

    return {t: float(w) for t, w in zip(tickers, weights)}