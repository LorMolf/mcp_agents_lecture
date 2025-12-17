#!/usr/bin/env python3
"""Quick test of supervisor system - non-interactive."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_analyst_system_supervisor import initialize_system, create_specialized_agents, build_supervisor_workflow, run_analysis

async def main():
    print("Testing Supervisor System...")

    # Initialize
    model, mcp_client, tools_by_category = await initialize_system()

    # Create agents
    agents = create_specialized_agents(model, tools_by_category)

    # Build supervisor workflow
    workflow = build_supervisor_workflow(agents, model)

    # Compile
    app = workflow.compile()
    print("Workflow compiled\n")

    # Test simple query
    await run_analysis(app, "What's the current price of Apple (AAPL)?")

    print("\nTest complete!")

if __name__ == "__main__":
    asyncio.run(main())
