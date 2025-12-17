from mcp.server.fastmcp import FastMCP
import yfinance as yf
import json
import random
import pandas as pd
from datetime import datetime, timedelta

mcp = FastMCP("StockData")

def generate_mock_history(period="1mo"):
    """Generate mock historical data for fallback."""
    dates = pd.date_range(end=datetime.now(), periods=30)
    base_price = 150.0
    prices = [base_price + random.uniform(-10, 10) for _ in range(30)]
    
    # Create simple structure matching yf output
    data = {
        "Close": prices,
        "High": [p + 2 for p in prices],
        "Low": [p - 2 for p in prices],
        "Volume": [int(random.uniform(1000000, 5000000)) for _ in range(30)]
    }
    return pd.DataFrame(data, index=dates)

@mcp.tool()
def get_stock_price(ticker: str) -> dict:
    """Get current stock price and daily trading information.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    
    Returns:
        Dict with current price, high, low, volume, and market cap
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        info = stock.info # This often triggers 429 too
        
        if hist.empty:
             raise ValueError("Empty data")
        
        return {
            "ticker": ticker,
            "current_price": round(float(hist["Close"].iloc[-1]), 2),
            "day_high": round(float(hist["High"].iloc[-1]), 2),
            "day_low": round(float(hist["Low"].iloc[-1]), 2),
            "volume": int(hist["Volume"].iloc[-1]),
            "market_cap": info.get("marketCap", "N/A")
        }
    except Exception as e:
        # Fallback to mock data
        return {
            "ticker": ticker,
            "current_price": 420.69,
            "day_high": 425.00,
            "day_low": 415.00,
            "volume": 1000000,
            "market_cap": 2500000000000,
            "note": "Mock data used due to API limits"
        }

@mcp.tool()
def get_historical_data(ticker: str, period: str = "1mo") -> str:
    """Get historical price data and calculate metrics.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '5y')
    
    Returns:
        JSON string with historical data and price changes
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
             raise ValueError("Empty data")
             
        return json.dumps({
            "ticker": ticker,
            "period": period,
            "start_price": round(float(hist['Close'].iloc[0]), 2),
            "end_price": round(float(hist['Close'].iloc[-1]), 2),
            "price_change_pct": round(float((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100), 2),
            "avg_price": round(float(hist['Close'].mean()), 2),
            "high": round(float(hist['High'].max()), 2),
            "low": round(float(hist['Low'].min()), 2)
        })
    except Exception as e:
        # Fallback
        return json.dumps({
            "ticker": ticker,
            "period": period,
            "start_price": 100.0,
            "end_price": 110.0,
            "price_change_pct": 10.0,
            "avg_price": 105.0,
            "high": 115.0,
            "low": 95.0,
            "note": "Mock data used due to API limits"
        })

@mcp.tool()
def get_stock_info(ticker: str) -> dict:
    """Get detailed company information for a stock.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    
    Returns:
        Dict with company name, sector, market cap, PE ratio, and beta
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ticker": ticker,
            "company_name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "beta": info.get("beta", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A")
        }
    except Exception:
        return {
            "ticker": ticker,
            "company_name": f"{ticker} Corp (Mock)",
            "sector": "Technology",
            "industry": "Software",
            "market_cap": 2000000000000,
            "pe_ratio": 35.5,
            "beta": 1.2,
            "52_week_high": 450.0,
            "52_week_low": 300.0,
            "note": "Mock data used due to API limits"
        }

if __name__ == "__main__":
    # Run as STDIO server for MCP client
    mcp.run(transport="stdio")
