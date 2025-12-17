"""
Demonstration of full workflow with step-by-step execution.
This will show each agent in action and generate actual files.
"""

import asyncio
from financial_analyst_system import initialize_system, build_graph
from langchain_core.messages import HumanMessage

async def run_workflow_demo():
    """Run a series of focused queries that will definitely produce outputs."""
    print("\n" + "="*80)
    print("🎬 FULL WORKFLOW DEMONSTRATION")
    print("="*80)
    print("\nRunning 4 separate queries to demonstrate each agent:\n")

    # Initialize
    print("⚙️  Initializing system...")
    model, mcp_client, tools_by_category = await initialize_system()
    graph = await build_graph(model, tools_by_category)
    print("✅ System ready!\n")

    # Query 1: Data Analyst
    print("\n" + "="*80)
    print("TEST 1: DATA ANALYST")
    print("="*80)
    query1 = "Get the current price, historical data for 6 months, and company info for Apple (AAPL)"
    print(f"Query: {query1}\n")

    config1 = {"configurable": {"thread_id": "demo_1"}, "recursion_limit": 20}
    input1 = {"messages": [HumanMessage(content=query1)], "next": "", "analysis_context": {}}

    print("Executing...\n")
    async for event in graph.astream(input1, config=config1):
        for node_name, node_output in event.items():
            if node_name != "__end__":
                print(f"→ {node_name.upper()}")
                if "messages" in node_output and node_output["messages"]:
                    msg = node_output["messages"][-1]
                    if hasattr(msg, 'content') and msg.content and node_name != "supervisor":
                        print(f"  {msg.content[:300]}")

    # Query 2: Chart Specialist
    print("\n" + "="*80)
    print("TEST 2: CHART SPECIALIST")
    print("="*80)
    query2 = "Create a comparison chart for AAPL and TSLA over 6 months"
    print(f"Query: {query2}\n")

    config2 = {"configurable": {"thread_id": "demo_2"}, "recursion_limit": 20}
    input2 = {"messages": [HumanMessage(content=query2)], "next": "", "analysis_context": {}}

    print("Executing...\n")
    async for event in graph.astream(input2, config=config2):
        for node_name, node_output in event.items():
            if node_name != "__end__":
                print(f"→ {node_name.upper()}")
                if "messages" in node_output and node_output["messages"]:
                    msg = node_output["messages"][-1]
                    if hasattr(msg, 'content') and msg.content and node_name != "supervisor":
                        print(f"  {msg.content[:300]}")

    # Query 3: News Analyst
    print("\n" + "="*80)
    print("TEST 3: NEWS ANALYST")
    print("="*80)
    query3 = "Get the latest news about Tesla (TSLA)"
    print(f"Query: {query3}\n")

    config3 = {"configurable": {"thread_id": "demo_3"}, "recursion_limit": 20}
    input3 = {"messages": [HumanMessage(content=query3)], "next": "", "analysis_context": {}}

    print("Executing...\n")
    async for event in graph.astream(input3, config=config3):
        for node_name, node_output in event.items():
            if node_name != "__end__":
                print(f"→ {node_name.upper()}")
                if "messages" in node_output and node_output["messages"]:
                    msg = node_output["messages"][-1]
                    if hasattr(msg, 'content') and msg.content and node_name != "supervisor":
                        print(f"  {msg.content[:300]}")

    # Query 4: Full Pipeline with Report
    print("\n" + "="*80)
    print("TEST 4: FULL PIPELINE (Data + Chart + Report)")
    print("="*80)
    query4 = """Analyze Tesla stock: get the price data for 3 months, create a chart,
    and save a report titled 'Tesla Analysis December 2025' with the findings."""
    print(f"Query: {query4}\n")

    config4 = {"configurable": {"thread_id": "demo_4"}, "recursion_limit": 30}
    input4 = {"messages": [HumanMessage(content=query4)], "next": "", "analysis_context": {}}

    print("Executing...\n")
    async for event in graph.astream(input4, config=config4):
        for node_name, node_output in event.items():
            if node_name != "__end__":
                print(f"→ {node_name.upper()}")
                if "messages" in node_output and node_output["messages"]:
                    msg = node_output["messages"][-1]
                    if hasattr(msg, 'content') and msg.content and node_name != "supervisor":
                        # Show more for report writer
                        max_len = 500 if node_name == "report_writer" else 300
                        print(f"  {msg.content[:max_len]}")

    # Final Summary
    print("\n" + "="*80)
    print("📊 DEMONSTRATION COMPLETE")
    print("="*80)

    import glob
    import os

    print("\n📁 Generated Files:\n")

    # Charts
    charts = sorted(glob.glob("*.png"), key=os.path.getmtime, reverse=True)
    if charts:
        print(f"  Charts ({len(charts)}):")
        for chart in charts[:5]:  # Show latest 5
            size = os.path.getsize(chart) / 1024
            mtime = os.path.getmtime(chart)
            from datetime import datetime
            timestamp = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
            print(f"    • {chart} ({size:.1f} KB) - created at {timestamp}")
    else:
        print(f"  ⚠️  No charts generated")

    # Reports
    reports = sorted(glob.glob("*.md"), key=os.path.getmtime, reverse=True)
    report_files = [r for r in reports if r not in ['README.md', 'TEST_RESULTS.md']]
    if report_files:
        print(f"\n  Reports ({len(report_files)}):")
        for report in report_files[:5]:  # Show latest 5
            size = os.path.getsize(report) / 1024
            mtime = os.path.getmtime(report)
            timestamp = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
            print(f"    • {report} ({size:.1f} KB) - created at {timestamp}")

            # Show first few lines of report
            print(f"\n      Preview of {report}:")
            with open(report, 'r') as f:
                lines = f.readlines()[:10]
                for line in lines:
                    print(f"      {line.rstrip()}")
            print()
    else:
        print(f"  ⚠️  No reports generated")

    print("\n" + "="*80)
    print("✅ All demonstrations completed!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(run_workflow_demo())
