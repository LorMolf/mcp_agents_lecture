"""
Complex query test to demonstrate full multi-agent pipeline.
This will generate data, charts, and a comprehensive report.
"""

import asyncio
from financial_analyst_system import initialize_system, build_graph
from langchain_core.messages import HumanMessage

async def run_complex_query():
    """Run a complex multi-step analysis query."""
    print("\n" + "="*80)
    print("🚀 COMPLEX QUERY TEST - Full Multi-Agent Pipeline")
    print("="*80)
    print("\nThis test will:")
    print("  1. Fetch data for multiple stocks")
    print("  2. Generate comparison charts")
    print("  3. Get latest news")
    print("  4. Create a comprehensive analysis report")
    print("\n" + "="*80)

    # Initialize
    print("\n⚙️  Initializing system...")
    model, mcp_client, tools_by_category = await initialize_system()
    graph = await build_graph(model, tools_by_category)
    print("✅ System initialized!\n")

    # Complex query that should trigger multiple agents
    query = """
    Perform a comprehensive analysis of Apple (AAPL) and Tesla (TSLA):

    1. Get the current stock prices and company information for both
    2. Get historical data for the last 6 months for both stocks
    3. Create a comparison chart showing their performance
    4. Get the latest news for both companies
    5. Save a detailed analysis report with all findings

    Make sure the report includes:
    - Current prices and key metrics
    - Historical performance comparison
    - Recent news summary
    - Overall analysis and insights
    """

    print(f"\n{'='*80}")
    print(f"📝 QUERY:")
    print(f"{'='*80}")
    print(query)
    print(f"{'='*80}\n")

    config = {
        "configurable": {"thread_id": "complex_analysis"},
        "recursion_limit": 40  # Higher limit for complex workflow
    }

    input_state = {
        "messages": [HumanMessage(content=query)],
        "next": "",
        "analysis_context": {}
    }

    agents_sequence = []
    message_count = 0

    try:
        print("🔄 Starting analysis...\n")

        async for event in graph.astream(input_state, config=config):
            for node_name, node_output in event.items():
                if node_name != "__end__":
                    agents_sequence.append(node_name)
                    message_count += 1

                    print(f"\n{'─'*80}")
                    print(f"📍 Step {message_count}: {node_name.upper()}")
                    print(f"{'─'*80}")

                    if "messages" in node_output and node_output["messages"]:
                        last_msg = node_output["messages"][-1]
                        if hasattr(last_msg, 'content') and last_msg.content:
                            content = last_msg.content

                            # Show full content for important messages
                            if node_name != "supervisor":
                                print(f"\n{content}\n")
                            else:
                                # For supervisor, just show routing decision
                                print(f"\n→ {content[:100]}\n")

        print(f"\n{'='*80}")
        print("✅ ANALYSIS COMPLETED!")
        print(f"{'='*80}")
        print(f"\n📊 Execution Summary:")
        print(f"   Total steps: {message_count}")
        print(f"   Agent sequence: {' → '.join(agents_sequence)}")

        # Count agent calls
        from collections import Counter
        agent_counts = Counter(agents_sequence)
        print(f"\n   Agent call counts:")
        for agent, count in agent_counts.most_common():
            print(f"      • {agent}: {count} calls")

        # Check for generated files
        print(f"\n📁 Generated Files:")
        import os
        import glob

        # Check for charts
        charts = glob.glob("*.png")
        if charts:
            print(f"   Charts:")
            for chart in charts[-3:]:  # Show last 3
                size = os.path.getsize(chart) / 1024
                print(f"      • {chart} ({size:.1f} KB)")
        else:
            print(f"   No charts found")

        # Check for reports
        reports = glob.glob("*.md")
        report_files = [r for r in reports if r not in ['README.md', 'TEST_RESULTS.md']]
        if report_files:
            print(f"   Reports:")
            for report in report_files[-3:]:  # Show last 3
                size = os.path.getsize(report) / 1024
                print(f"      • {report} ({size:.1f} KB)")
        else:
            print(f"   No reports found")

        print(f"\n{'='*80}")
        print("🎉 SUCCESS! Review the generated files above.")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_complex_query())
