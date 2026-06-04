"""Fetch current market context (weather, news, trends) for supply chain decisions.

This module uses the repo MCP servers where available:
- `mcp.weather_server` for weather
- `mcp.news_server` for headlines
- `mcp.finance_server` for market/commodity prices
"""

from typing import Dict, List
from datetime import datetime

from mcp import get_weather_mcp, get_news_mcp, get_finance_mcp


def build_market_context_prompt(product_category: str, month: int) -> str:
    """Build a market context string for the LLM prompt using MCP data.

    Falls back to lightweight defaults when MCPs are not available.
    """
    now = datetime.now()
    season = get_season(month)

    # Weather via MCP
    weather_mcp = get_weather_mcp()
    # default coords (New York) if geocoding not provided
    try:
        coords = weather_mcp.get_coordinates("New York") or {"latitude": 40.7128, "longitude": -74.0060}
    except Exception:
        coords = {"latitude": 40.7128, "longitude": -74.0060}

    weather = weather_mcp.get_current_weather(coords["latitude"], coords["longitude"]) or {}
    weather_str = "Weather data unavailable"
    if weather and weather.get("current"):
        cur = weather["current"]
        weather_str = f"[Weather] Current weather at New York: {cur.get('temperature_2m')}°C, Humidity: {cur.get('relative_humidity_2m')}%, Wind: {cur.get('wind_speed_10m')}km/h"

    # News via MCP
    news_mcp = get_news_mcp()
    try:
        headlines = news_mcp["get_headlines"](f"{product_category} supply chain", page_size=3)
    except Exception:
        headlines = ["No news available"]
    news_str = "[News] Recent market headlines:\n" + "\n".join([f"- {h}" for h in headlines[:3]])

    # Finance via MCP (example: crude oil futures symbol)
    finance_mcp = get_finance_mcp()
    oil_price = None
    try:
        fin = finance_mcp["get_latest_price"]("CL=F")
        if fin and fin.get("price"):
            oil_price = fin["price"]
    except Exception:
        oil_price = None

    finance_str = f"[Finance] Crude oil latest price: ${oil_price}" if oil_price else "[Finance] Oil price unavailable"

    context = f"""
Market Context (as of {now.strftime('%Y-%m-%d')}):
- Season: {season}
- Product Category: {product_category}

{weather_str}

{news_str}

{finance_str}

Additional global conditions to consider:
- Geopolitical tensions affecting shipping routes
- Port congestion levels in major hubs
- Seasonal demand patterns
- Currency fluctuations affecting import/export costs
"""
    return context


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"
