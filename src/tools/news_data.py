"""
News fetching. Uses yfinance's built-in news feed so no separate API key
is required to get started. Swap this out for Finnhub later if you want
richer sentiment data - the function signature would stay the same, so
nothing else in the codebase needs to change.
"""
import yfinance as yf


def get_recent_headlines(ticker: str, max_headlines: int = 3) -> str:
    """
    Fetch a handful of recent headlines for a ticker and return them
    as a short bulleted string for prompt injection.
    """
    try:
        news_items = yf.Ticker(ticker).news[:max_headlines]
    except Exception:
        return f"No recent news available for {ticker}."

    if not news_items:
        return f"No recent news available for {ticker}."

    lines = []
    for item in news_items:
        # yfinance nests the actual headline under a "content" key in
        # recent versions of the library - handle both shapes defensively.
        content = item.get("content", item)
        title = content.get("title", "Untitled")
        lines.append(f"- {title}")

    return f"Recent headlines for {ticker}:\n" + "\n".join(lines)
