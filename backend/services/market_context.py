# services/market_context.py

try:
    from mcp.news_server import get_news_mcp
    from mcp.finance_server import get_finance_mcp
except ImportError as e:
    print(f"Market context MCP import warning: {e}")
    get_news_mcp = None
    get_finance_mcp = None

def build_market_context_prompt(product_category: str, month: int) -> str:
    """
    Gathers real-time news and financial data from MCPs to provide 
    market context to the Groq LLM.
    """
    context_lines = []
    
    # 1. Fetch News Context
    if get_news_mcp:
        try:
            news_mcp = get_news_mcp()
            get_headlines = news_mcp.get("get_headlines")
            if get_headlines:
                headlines = get_headlines(f"{product_category} logistics supply chain", page_size=2)
                context_lines.append(f"[News] Recent headlines: {', '.join(headlines)}")
        except Exception as e:
            print(f"Failed to fetch news: {e}")
            
    # 2. Fetch Finance Context (Using Crude Oil 'CL=F' as a standard freight cost indicator)
    if get_finance_mcp:
        try:
            finance_mcp = get_finance_mcp()
            get_price = finance_mcp.get("get_latest_price")
            if get_price:
                oil_data = get_price("CL=F")
                if oil_data and "price" in oil_data:
                    context_lines.append(f"[Finance] Current Crude Oil (WTI) price: ${oil_data['price']:.2f}")
        except Exception as e:
            print(f"Failed to fetch finance data: {e}")

    # 3. Add Seasonal awareness based on the month
    season = "Winter" if month in [12, 1, 2] else "Spring" if month in [3, 4, 5] else "Summer" if month in [6, 7, 8] else "Fall"
    context_lines.append(f"[Season] Order is placed for month {month} ({season}). Factor in typical seasonal weather disruptions for this time of year.")

    # Combine it all into a single string for the prompt
    if not context_lines:
        return "No real-time market data available. Rely on standard supply chain principles."
        
    return "\n".join(context_lines)