"""MCP Finance Server
Provides market/commodity price data using yfinance (no API key required).
"""
from typing import Optional
import os

try:
    import yfinance as yf
except Exception:
    yf = None


def get_latest_price(ticker: str) -> Optional[dict]:
    """Return latest market price info for a given ticker using yfinance.
    Returns dict with price, currency, timestamp or None if unavailable.
    """
    if yf is None:
        print("yfinance not installed; finance MCP unavailable.")
        return None
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1d", interval="1m")
        if hist is None or hist.empty:
            info = tk.info if hasattr(tk, 'info') else {}
            return {
                "price": info.get("regularMarketPrice"),
                "currency": info.get("currency"),
                "symbol": ticker,
            }
        latest = hist.iloc[-1]
        return {
            "price": float(latest['Close']),
            "timestamp": str(latest.name),
            "symbol": ticker,
        }
    except Exception as e:
        print(f"Finance MCP error for {ticker}: {e}")
        return None


# Singleton accessor
_finance_mcp = None

def get_finance_mcp():
    global _finance_mcp
    if _finance_mcp is None:
        _finance_mcp = True  # placeholder flag; functions are module-level
    return {
        "get_latest_price": get_latest_price
    }
