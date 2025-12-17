#!/usr/bin/env python3
"""Comprehensive test - create a full analysis report."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_analyst_system_supervisor import initialize_system, create_specialized_agents, build_supervisor_workflow, run_analysis

async def main():
    print("=== COMPREHENSIVE TEST ===\n")

    # Initialize
    model, mcp_client, tools_by_category = await initialize_system()

    # Create agents
    agents = create_specialized_agents(model, tools_by_category)

    # Build supervisor workflow
    workflow = build_supervisor_workflow(agents, model)

    # Compile
    app = workflow.compile()
    print("Workflow compiled\n")

    # Comprehensive query
    query = """Analyze Apple (AAPL) stock:
1. Get current stock price and company information
2. Create a 3-month price chart
3. Get recent news
4. Save a comprehensive financial analysis report

Make sure to complete ALL steps."""

    await run_analysis(app, query)

    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(main())
