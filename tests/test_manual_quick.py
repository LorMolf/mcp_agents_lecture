#!/usr/bin/env python3
"""Quick test of manual graph system - non-interactive."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_analyst_system_manual_graph import initialize_system, build_graph, run_analysis

async def main():
    print("Testing Manual Graph System...")

    # Initialize
    model, mcp_client, tools_by_category = await initialize_system()

    # Build graph
    graph = await build_graph(model, tools_by_category)

    # Test simple query
    await run_analysis(graph, "What's the current price of Apple (AAPL)?", thread_id="test1")

    print("\nTest complete!")

if __name__ == "__main__":
    asyncio.run(main())
