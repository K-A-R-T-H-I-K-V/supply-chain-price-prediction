"""MCP News Server
Wraps NewsAPI (or fallback) to provide recent headlines for market context.
"""
import os
import requests
from typing import List, Optional

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def get_headlines(query: str = "supply chain", page_size: int = 3) -> List[str]:
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        # Fallback suggestions if key missing
        return [
            "Monitor global shipping rates and container indices.",
            "Watch for port congestions and labor actions.",
            "Track commodity price movements affecting input costs.",
        ]
    try:
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": page_size,
            "apiKey": api_key,
        }
        resp = requests.get(NEWSAPI_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        headlines = [f"{a.get('title')} ({a.get('source', {}).get('name')})" for a in articles[:page_size]]
        return headlines or ["No recent news found"]
    except Exception as e:
        print(f"News MCP error: {e}")
        return ["Unable to fetch news at this time"]


# Singleton accessor
_news_mcp = None

def get_news_mcp():
    global _news_mcp
    if _news_mcp is None:
        _news_mcp = True
    return {"get_headlines": get_headlines}
