"""
Pipeline Comparison Script

Runs the same query on both orchestration methods and compares:
- Execution time
- Agent routing sequence
- Output quality
- Token usage (if available)
- Generated artifacts

Usage:
    python scripts/compare_orchestrations.py "Your query here"
    python scripts/compare_orchestrations.py --interactive
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage

# Import both systems
print("Loading orchestration systems...")
try:
    from financial_analyst_system_manual_graph import (
        initialize_system as init_manual,
        build_graph as build_manual_graph
    )
    MANUAL_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Manual graph system not available: {e}")
    MANUAL_AVAILABLE = False

try:
    from financial_analyst_system_supervisor import (
        initialize_system as init_supervisor,
        create_specialized_agents,
        build_supervisor_workflow
    )
    SUPERVISOR_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Supervisor system not available: {e}")
    SUPERVISOR_AVAILABLE = False

# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

class ExecutionMetrics:
    """Track execution metrics for a pipeline run."""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.agents_called = []
        self.steps = 0
        self.errors = []
        self.outputs = []

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

    def add_agent(self, agent_name: str):
        self.agents_called.append(agent_name)
        self.steps += 1

    def add_error(self, error: str):
        self.errors.append(error)

    def add_output(self, output: str):
        self.outputs.append(output)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "duration": f"{self.duration:.2f}s" if self.duration else "N/A",
            "steps": self.steps,
            "agents": list(set(self.agents_called)),
            "agent_sequence": " → ".join(self.agents_called),
            "errors": len(self.errors),
            "success": len(self.errors) == 0
        }

# ============================================================================
# MANUAL GRAPH PIPELINE
# ============================================================================

async def run_manual_graph(query: str) -> ExecutionMetrics:
    """Run query on manual graph orchestration."""
    metrics = ExecutionMetrics("Manual Graph")

    if not MANUAL_AVAILABLE:
        metrics.add_error("Manual graph system not available")
        return metrics

    try:
        print("\n" + "="*80)
        print("🔧 MANUAL GRAPH ORCHESTRATION")
        print("="*80)

        metrics.start()

        # Initialize
        model, mcp_client, tools_by_category = await init_manual()
        graph = await build_manual_graph(model, tools_by_category)

        # Run query
        config = {
            "configurable": {"thread_id": f"manual_{int(time.time())}"},
            "recursion_limit": 30
        }

        input_state = {
            "messages": [HumanMessage(content=query)],
            "next": "",
            "analysis_context": {}
        }

        async for event in graph.astream(input_state, config=config):
            for node_name, node_output in event.items():
                if node_name != "__end__":
                    metrics.add_agent(node_name)
                    print(f"  → {node_name}")

                    if "messages" in node_output and node_output["messages"]:
                        last_msg = node_output["messages"][-1]
                        if hasattr(last_msg, 'content') and last_msg.content:
                            content = last_msg.content[:150]
                            print(f"     {content}...")
                            if len(last_msg.content) > 150:
                                metrics.add_output(last_msg.content)

        metrics.end()

    except Exception as e:
        metrics.add_error(str(e))
        print(f"  ❌ Error: {e}")

    return metrics

# ============================================================================
# SUPERVISOR PIPELINE
# ============================================================================

async def run_supervisor(query: str) -> ExecutionMetrics:
    """Run query on supervisor orchestration."""
    metrics = ExecutionMetrics("Supervisor Pattern")

    if not SUPERVISOR_AVAILABLE:
        metrics.add_error("Supervisor system not available")
        return metrics

    try:
        print("\n" + "="*80)
        print("🎯 SUPERVISOR ORCHESTRATION")
        print("="*80)

        metrics.start()

        # Initialize
        model, mcp_client, tools_by_category = await init_supervisor()
        agents = create_specialized_agents(model, tools_by_category)
        workflow = build_supervisor_workflow(agents, model)
        app = workflow.compile()

        # Run query
        result = app.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        })

        # Extract agent calls from result
        if "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, 'name') and msg.name:
                    metrics.add_agent(msg.name)
                    print(f"  → {msg.name}")

                if hasattr(msg, 'content') and msg.content:
                    content_preview = msg.content[:150]
                    print(f"     {content_preview}...")
                    if len(msg.content) > 150:
                        metrics.add_output(msg.content)

        metrics.end()

    except Exception as e:
        metrics.add_error(str(e))
        print(f"  ❌ Error: {e}")

    return metrics

# ============================================================================
# COMPARISON & REPORTING
# ============================================================================

def print_comparison(manual_metrics: ExecutionMetrics, supervisor_metrics: ExecutionMetrics):
    """Print detailed comparison of both runs."""

    print("\n" + "="*80)
    print("📊 COMPARISON RESULTS")
    print("="*80)

    # Execution Time
    print("\n⏱️  Execution Time:")
    print(f"  Manual Graph:  {manual_metrics.duration:.2f}s" if manual_metrics.duration else "  Manual Graph:  Failed")
    print(f"  Supervisor:    {supervisor_metrics.duration:.2f}s" if supervisor_metrics.duration else "  Supervisor:    Failed")

    if manual_metrics.duration and supervisor_metrics.duration:
        diff = abs(manual_metrics.duration - supervisor_metrics.duration)
        faster = "Manual Graph" if manual_metrics.duration < supervisor_metrics.duration else "Supervisor"
        percent = (diff / max(manual_metrics.duration, supervisor_metrics.duration)) * 100
        print(f"  ⚡ {faster} was {diff:.2f}s ({percent:.1f}%) faster")

    # Steps and Agent Calls
    print("\n📍 Workflow Steps:")
    print(f"  Manual Graph:  {manual_metrics.steps} steps")
    print(f"  Supervisor:    {supervisor_metrics.steps} steps")

    # Agent Sequence
    print("\n🔄 Agent Call Sequence:")
    print(f"  Manual Graph:")
    if manual_metrics.agents_called:
        print(f"    {' → '.join(manual_metrics.agents_called)}")
    else:
        print(f"    No agents called")

    print(f"  Supervisor:")
    if supervisor_metrics.agents_called:
        print(f"    {' → '.join(supervisor_metrics.agents_called)}")
    else:
        print(f"    No agents called")

    # Unique Agents
    print("\n🤖 Unique Agents Used:")
    manual_agents = set(manual_metrics.agents_called)
    supervisor_agents = set(supervisor_metrics.agents_called)

    print(f"  Manual Graph:  {', '.join(sorted(manual_agents)) if manual_agents else 'None'}")
    print(f"  Supervisor:    {', '.join(sorted(supervisor_agents)) if supervisor_agents else 'None'}")

    # Differences
    only_manual = manual_agents - supervisor_agents
    only_supervisor = supervisor_agents - manual_agents
    if only_manual:
        print(f"  ℹ️  Only in Manual: {', '.join(only_manual)}")
    if only_supervisor:
        print(f"  ℹ️  Only in Supervisor: {', '.join(only_supervisor)}")

    # Errors
    print("\n⚠️  Errors:")
    print(f"  Manual Graph:  {len(manual_metrics.errors)} errors")
    print(f"  Supervisor:    {len(supervisor_metrics.errors)} errors")

    if manual_metrics.errors:
        for i, error in enumerate(manual_metrics.errors, 1):
            print(f"    {i}. {error}")
    if supervisor_metrics.errors:
        for i, error in enumerate(supervisor_metrics.errors, 1):
            print(f"    {i}. {error}")

    # Success
    print("\n✅ Success:")
    print(f"  Manual Graph:  {'✓' if manual_metrics.summary()['success'] else '✗'}")
    print(f"  Supervisor:    {'✓' if supervisor_metrics.summary()['success'] else '✗'}")

    # Check generated files
    print("\n📁 Generated Outputs:")
    check_generated_files()

    # Overall Assessment
    print("\n" + "="*80)
    print("🎯 ASSESSMENT")
    print("="*80)

    if manual_metrics.summary()['success'] and supervisor_metrics.summary()['success']:
        print("✅ Both pipelines completed successfully")

        if manual_metrics.agents_called == supervisor_metrics.agents_called:
            print("✅ Identical agent routing")
        else:
            print("ℹ️  Different agent routing patterns")

        if manual_metrics.duration and supervisor_metrics.duration:
            if abs(manual_metrics.duration - supervisor_metrics.duration) < 2:
                print("✅ Similar performance (~same speed)")
            else:
                print(f"ℹ️  Performance difference: {abs(manual_metrics.duration - supervisor_metrics.duration):.2f}s")
    else:
        if not manual_metrics.summary()['success']:
            print("❌ Manual Graph pipeline failed")
        if not supervisor_metrics.summary()['success']:
            print("❌ Supervisor pipeline failed")

    print("="*80)

def check_generated_files():
    """Check for recently generated files."""
    outputs_dir = Path(__file__).parent.parent / "outputs"

    # Check charts
    charts_dir = outputs_dir / "charts"
    if charts_dir.exists():
        charts = sorted(charts_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        recent_charts = [c for c in charts if (time.time() - c.stat().st_mtime) < 300]  # Last 5 min
        if recent_charts:
            print(f"  Charts (generated in last 5 min): {len(recent_charts)}")
            for chart in recent_charts[:3]:
                print(f"    • {chart.name}")
        else:
            print(f"  Charts: No recent files")
    else:
        print(f"  Charts: Directory not found")

    # Check reports
    reports_dir = outputs_dir / "reports"
    if reports_dir.exists():
        reports = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        recent_reports = [r for r in reports if (time.time() - r.stat().st_mtime) < 300]
        if recent_reports:
            print(f"  Reports (generated in last 5 min): {len(recent_reports)}")
            for report in recent_reports[:3]:
                print(f"    • {report.name}")
        else:
            print(f"  Reports: No recent files")
    else:
        print(f"  Reports: Directory not found")

# ============================================================================
# MAIN
# ============================================================================

async def compare_pipelines(query: str):
    """Run the same query on both pipelines and compare."""

    print("\n" + "="*80)
    print("🔬 ORCHESTRATION COMPARISON")
    print("="*80)
    print(f"\n📝 Query: {query}")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check availability
    if not MANUAL_AVAILABLE and not SUPERVISOR_AVAILABLE:
        print("\n❌ Neither pipeline is available!")
        return

    if not MANUAL_AVAILABLE:
        print("\n⚠️  Manual graph pipeline not available, running supervisor only...")
        supervisor_metrics = await run_supervisor(query)
        print_comparison(ExecutionMetrics("Manual Graph"), supervisor_metrics)
        return

    if not SUPERVISOR_AVAILABLE:
        print("\n⚠️  Supervisor pipeline not available, running manual graph only...")
        manual_metrics = await run_manual_graph(query)
        print_comparison(manual_metrics, ExecutionMetrics("Supervisor Pattern"))
        return

    # Run both
    print("\n🚀 Running on both orchestration methods...")

    # Run Manual Graph
    manual_metrics = await run_manual_graph(query)

    # Wait a moment between runs
    await asyncio.sleep(1)

    # Run Supervisor
    supervisor_metrics = await run_supervisor(query)

    # Compare results
    print_comparison(manual_metrics, supervisor_metrics)

async def interactive_mode():
    """Interactive comparison mode."""
    print("\n" + "="*80)
    print("🔬 Interactive Comparison Mode")
    print("="*80)
    print("\nCompare how both orchestration methods handle queries.")
    print("\nCommands:")
    print("  • Enter query to compare both pipelines")
    print("  • 'examples' for example queries")
    print("  • 'quit' to exit")
    print("\n" + "="*80)

    while True:
        try:
            query = input("\n💬 Query to compare: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if query.lower() == 'examples':
                print("\n📋 Example Queries:")
                print("  1. What's Apple's current stock price?")
                print("  2. Create a chart for Tesla over 6 months")
                print("  3. Compare AAPL and MSFT")
                print("  4. Analyze Microsoft and create a report")
                continue

            if not query:
                continue

            await compare_pipelines(query)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Compare orchestration methods")
    parser.add_argument("query", nargs="?", help="Query to run on both pipelines")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    if args.interactive:
        await interactive_mode()
    elif args.query:
        await compare_pipelines(args.query)
    else:
        # Default: run a test query
        default_query = "What's the current price of Apple (AAPL)?"
        print(f"Running default query: {default_query}")
        print("(Use -i for interactive mode or provide a query as argument)")
        await compare_pipelines(default_query)

if __name__ == "__main__":
    asyncio.run(main())
