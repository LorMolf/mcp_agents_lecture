from mcp.server.fastmcp import FastMCP
import yfinance as yf
import json
from datetime import datetime

mcp = FastMCP("News")

@mcp.tool()
def get_stock_news(ticker: str, limit: int = 10) -> str:
    """Get recent news articles for a stock ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        limit: Maximum number of articles to retrieve (default 10)
    
    Returns:
        JSON string with news articles
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:limit]
        articles = [{
            "title": item.get("title", "N/A"),
            "publisher": item.get("publisher", "N/A"),
            "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d") if item.get("providerPublishTime") else "N/A"
        } for item in news]
        return json.dumps({"ticker": ticker, "articles": articles}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    # Run as STDIO server for MCP client
    mcp.run(transport="stdio")
