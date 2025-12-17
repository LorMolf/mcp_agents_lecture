from mcp.server.fastmcp import FastMCP
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

mcp = FastMCP("Plot")
sns.set_style("darkgrid")

# Output directory for charts
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_mock_data(period="3mo"):
    """Generate mock price data."""
    dates = pd.date_range(end=datetime.now(), periods=90)
    prices = np.cumsum(np.random.randn(90)) + 100
    return pd.DataFrame({"Close": prices}, index=dates)

@mcp.tool()
def create_chart(ticker: str, period: str = "3mo") -> dict:
    """Create a price chart for a stock ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '5y')
    
    Returns:
        Dict with success status and filename
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
             # Force error to trigger fallback
             raise ValueError("Empty data")
        
        plt.figure(figsize=(10, 5))
        plt.plot(hist.index, hist['Close'], linewidth=2, color='blue')
        plt.title(f"{ticker} - {period}", fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{ticker}_{period}.png"
        filepath = OUTPUT_DIR / filename
        plt.savefig(filepath, dpi=120)
        plt.close()
        return {"success": True, "filename": str(filepath), "message": f"Chart saved to {filepath}"}
    except Exception as e:
        # Fallback Mock Plot
        plt.close()
        try:
            hist = generate_mock_data(period)
            plt.figure(figsize=(10, 5))
            plt.plot(hist.index, hist['Close'], linewidth=2, color='green', linestyle="--")
            plt.title(f"{ticker} - {period} (MOCK DATA)", fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = f"{ticker}_{period}.png"
            filepath = OUTPUT_DIR / filename
            plt.savefig(filepath, dpi=120)
            plt.close()
            return {"success": True, "filename": str(filepath), "message": f"Chart saved to {filepath} (Mock Data)"}
        except Exception as e2:
             return {"success": False, "error": str(e)}

@mcp.tool()
def create_comparison(tickers: str, period: str = "3mo") -> dict:
    """Create a comparison chart for multiple stock tickers (normalized).
    
    Args:
        tickers: Comma-separated ticker symbols (e.g., 'AAPL,GOOGL,MSFT')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '5y')
    
    Returns:
        Dict with success status and filename
    """
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        plt.figure(figsize=(10, 5))
        
        for ticker in ticker_list:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                normalized = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
                plt.plot(hist.index, normalized, linewidth=2, label=ticker)
            else:
                raise ValueError("Empty data")
        
        plt.title(f"Comparison - {period}", fontsize=14, fontweight='bold')
        plt.ylabel('Change (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.tight_layout()
        
        filename = f"comparison_{'_'.join(ticker_list)}_{period}.png"
        filepath = OUTPUT_DIR / filename
        plt.savefig(filepath, dpi=120)
        plt.close()
        return {"success": True, "filename": str(filepath), "message": f"Comparison chart saved to {filepath}"}
    except Exception as e:
        # Fallback
        plt.close()
        try:
            ticker_list = [t.strip().upper() for t in tickers.split(',')]
            plt.figure(figsize=(10, 5))
            
            for ticker in ticker_list:
                hist = generate_mock_data(period)
                normalized = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
                plt.plot(hist.index, normalized, linewidth=2, label=f"{ticker} (Mock)")

            plt.title(f"Comparison - {period} (MOCK)", fontsize=14, fontweight='bold')
            plt.ylabel('Change (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
            plt.tight_layout()
            
            filename = f"comparison_{'_'.join(ticker_list)}_{period}.png"
            filepath = OUTPUT_DIR / filename
            plt.savefig(filepath, dpi=120)
            plt.close()
            return {"success": True, "filename": str(filepath), "message": f"Comparison chart saved to {filepath} (Mock Data)"}
        except Exception as e2:
            return {"success": False, "error": str(e2)}

if __name__ == "__main__":
    # Run as STDIO server for MCP client
    mcp.run(transport="stdio")
