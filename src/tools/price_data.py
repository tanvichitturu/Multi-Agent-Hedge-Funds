"""
Real market data fetching. Kept separate from agent/LLM code so it can be
unit-tested on its own and reused later by the backtest loop (Phase G).
"""
import yfinance as yf


def get_price_summary(ticker: str, period: str = "5d") -> str:
    """
    Fetch recent price history and return a short natural-language summary
    suitable for injecting into an LLM prompt.
    """
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        return f"No price data available for {ticker}."

    first_open = data["Open"].iloc[0]
    last_close = data["Close"].iloc[-1]
    pct_change = (last_close - first_open) / first_open * 100
    direction = "up" if pct_change >= 0 else "down"

    high = data["High"].max()
    low = data["Low"].min()

    return (
        f"{ticker} over the last {period}: opened at ${first_open:.2f}, "
        f"closed at ${last_close:.2f} ({direction} {abs(pct_change):.2f}%). "
        f"Period high: ${high:.2f}, period low: ${low:.2f}."
    )
def summarize_price_slice(closes, ticker: str) -> str:
    """
    Same idea as get_price_summary, but operates on an ALREADY-FETCHED
    slice of closing prices instead of calling yfinance live. This is
    what lets the backtest reason using only data that would have been
    available at each simulated point in time - pass it prices.iloc[:t+1]
    and it has zero visibility into anything after day t.
    """
    if len(closes) < 2:
        return f"Not enough price history available for {ticker}."

    first_open = closes.iloc[0]
    last_close = closes.iloc[-1]
    pct_change = (last_close - first_open) / first_open * 100
    direction = "up" if pct_change >= 0 else "down"
    high = closes.max()
    low = closes.min()

    return (
        f"{ticker} recent price action: started near ${first_open:.2f}, "
        f"now at ${last_close:.2f} ({direction} {abs(pct_change):.2f}%). "
        f"Recent high: ${high:.2f}, recent low: ${low:.2f}."
    )
def _check_volatility_trigger(closes) -> dict:
    """
    Pure logic, no data fetching - takes a pandas Series of closing prices
    and checks the paper's Extreme Market Conference trigger: a single day
    moving >5%, or three days cumulatively moving >10%. Kept separate from
    detect_extreme_volatility() so it can be unit tested with synthetic
    price series instead of needing a live network call.
    """
    daily_returns = closes.pct_change().dropna()
    max_daily_move = float(daily_returns.abs().max()) if not daily_returns.empty else 0.0

    three_day_move = 0.0
    if len(closes) >= 4:
        three_day_move = float((closes.iloc[-1] - closes.iloc[-4]) / closes.iloc[-4])

    daily_triggered = max_daily_move > 0.05
    three_day_triggered = abs(three_day_move) > 0.10

    return {
        "triggered": daily_triggered or three_day_triggered,
        "max_daily_move_pct": max_daily_move * 100,
        "three_day_move_pct": three_day_move * 100,
        "daily_triggered": daily_triggered,
        "three_day_triggered": three_day_triggered,
    }


def detect_extreme_volatility(ticker: str) -> dict:
    """
    Fetch recent price history for one ticker and check it against the
    Extreme Market Conference trigger (see _check_volatility_trigger).
    Returns a dict; result["triggered"] is what callers care about most.
    """
    data = yf.Ticker(ticker).history(period="5d")
    if data.empty or len(data) < 2:
        return {"triggered": False, "max_daily_move_pct": 0.0, "three_day_move_pct": 0.0,
                "daily_triggered": False, "three_day_triggered": False}

    result = _check_volatility_trigger(data["Close"])
    result["ticker"] = ticker
    return result
