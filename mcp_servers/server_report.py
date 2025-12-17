from mcp.server.fastmcp import FastMCP
from datetime import datetime
from pathlib import Path

mcp = FastMCP("Report")

# Output directory for reports
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@mcp.tool()
def save_report(title: str, content: str) -> dict:
    """Save a financial analysis report to a markdown file.
    
    Args:
        title: Report title
        content: Report content in markdown format
    
    Returns:
        Dict with success status and filename
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title).replace(' ', '_')[:50]
        filename = f"{safe_title}_{timestamp}.md"
        filepath = OUTPUT_DIR / filename
        full_content = f"""# {title}\n\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n{content}\n"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
        return {"success": True, "filename": str(filepath), "message": f"Report saved to {filepath}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Run as STDIO server for MCP client
    mcp.run(transport="stdio")
