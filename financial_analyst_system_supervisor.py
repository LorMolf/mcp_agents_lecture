"""
Financial Analyst Multi-Agent System (Version 2)

Alternative orchestration using langgraph_supervisor pattern.
This version uses a simpler supervisor creation method similar to orchestrator_option2.py

Architecture:
- Uses create_supervisor() for cleaner agent orchestration
- Maintains same specialized agents and MCP servers
- Simplified compared to v1's manual graph construction
"""

import asyncio
import sys
from pathlib import Path
from typing import TypedDict, Annotated

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# Try to import langgraph_supervisor (might need installation)
try:
    from langgraph_supervisor import create_supervisor
    HAS_SUPERVISOR = True
except ImportError:
    print("WARNING: langgraph_supervisor not found. Install with: pip install langgraph-supervisor")
    HAS_SUPERVISOR = False

import operator

# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_MODEL = "granite4:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Get absolute paths for MCP servers
SERVER_DIR = Path(__file__).parent / "mcp_servers"

# MCP Server configurations
MCP_SERVERS = {
    "stock_data": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER_DIR / "server_stock_data.py")],
    },
    "plot": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER_DIR / "server_plot.py")],
    },
    "news": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER_DIR / "server_news.py")],
    },
    "report": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER_DIR / "server_report.py")],
    }
}

# ============================================================================
# AGENT PROMPTS
# ============================================================================

DATA_ANALYST_PROMPT = """You are a Data Analyst. You MUST use tools for ALL stock data.

When asked about "Apple (AAPL)" stock:
- Call: get_stock_price(ticker="AAPL")
- Call: get_historical_data(ticker="AAPL", period="3mo")
- Call: get_stock_info(ticker="AAPL")

ALWAYS include the ticker parameter in EVERY tool call.
Example: get_historical_data(ticker="AAPL", period="3mo")
NOT: get_historical_data(period="3mo")

Use the tool NOW."""

CHART_SPECIALIST_PROMPT = """You are a Chart Specialist. You MUST use tools to create charts.

When asked to create a chart for "Apple (AAPL)":
- Call: create_chart(ticker="AAPL", period="3mo")

ALWAYS include ticker AND period in EVERY tool call.
Example: create_chart(ticker="AAPL", period="3mo")
NOT: create_chart(period="3mo")

Use the tool NOW. Charts save to outputs/charts/."""

NEWS_ANALYST_PROMPT = """You are a News Analyst. You MUST use tools for ALL news.

When asked about news for "Apple (AAPL)":
- Call: get_stock_news(ticker="AAPL")

ALWAYS include the ticker parameter.
Example: get_stock_news(ticker="AAPL")
NOT: get_stock_news()

Use the tool NOW."""

REPORT_WRITER_PROMPT = """You are a Report Writer. You MUST use tools to save reports.

Compile all conversation data into a report, then:
- Call: save_report(report_title="Apple Stock Analysis", report_content="[full report here]")

ALWAYS include report_title AND report_content parameters.
Example: save_report(report_title="AAPL Analysis", report_content="Price: $100...")

Use the tool NOW. Reports save to outputs/reports/."""

# ============================================================================
# INITIALIZATION
# ============================================================================

async def initialize_system():
    """Initialize Ollama model and MCP client."""
    print("Initializing Financial Analyst System (Version 2)...")

    # Initialize Ollama model
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )
    print(f"Ollama model '{OLLAMA_MODEL}' initialized")

    # Initialize MCP client
    print("Connecting to MCP servers...")
    mcp_client = MultiServerMCPClient(MCP_SERVERS)

    # Get all tools
    all_tools = await mcp_client.get_tools()
    print(f"Loaded {len(all_tools)} tools from {len(MCP_SERVERS)} servers")

    # Categorize tools
    tools_by_category = {
        "data": [t for t in all_tools if any(name in t.name for name in ["stock_price", "historical_data", "stock_info"])],
        "chart": [t for t in all_tools if any(name in t.name for name in ["create_chart", "create_comparison"])],
        "news": [t for t in all_tools if "news" in t.name],
        "report": [t for t in all_tools if "report" in t.name],
    }

    for category, tools in tools_by_category.items():
        print(f"  - {category}: {[t.name for t in tools]}")

    return model, mcp_client, tools_by_category

# ============================================================================
# AGENT CREATION
# ============================================================================

def create_specialized_agents(model, tools_by_category):
    """Create specialized agents using create_agent."""
    print("\nCreating specialized agents...")

    agents = []

    # Data Analyst
    data_analyst = create_agent(
        model=model,
        tools=tools_by_category["data"],
        system_prompt=DATA_ANALYST_PROMPT,
        name="data_analyst"
    )
    agents.append(data_analyst)
    print("  - Data Analyst")

    # Chart Specialist
    chart_specialist = create_agent(
        model=model,
        tools=tools_by_category["chart"],
        system_prompt=CHART_SPECIALIST_PROMPT,
        name="chart_specialist"
    )
    agents.append(chart_specialist)
    print("  - Chart Specialist")

    # News Analyst
    news_analyst = create_agent(
        model=model,
        tools=tools_by_category["news"],
        system_prompt=NEWS_ANALYST_PROMPT,
        name="news_analyst"
    )
    agents.append(news_analyst)
    print("  - News Analyst")

    # Report Writer
    report_writer = create_agent(
        model=model,
        tools=tools_by_category["report"],
        system_prompt=REPORT_WRITER_PROMPT,
        name="report_writer"
    )
    agents.append(report_writer)
    print("  - Report Writer")

    return agents

# ============================================================================
# SUPERVISOR WORKFLOW
# ============================================================================

def build_supervisor_workflow(agents, model):
    """Build supervisor workflow using create_supervisor."""
    if not HAS_SUPERVISOR:
        raise ImportError("langgraph_supervisor not available. This version requires it.")

    print("\nBuilding supervisor workflow...")

    # Supervisor prompt that forces delegation
    supervisor_prompt = """You are a supervisor coordinating specialist agents. You have NO tools yourself.

CRITICAL: You MUST delegate ALL tasks to your specialist agents:
- data_analyst: For stock prices, historical data, company information
- chart_specialist: For creating charts and visualizations
- news_analyst: For fetching financial news
- report_writer: For saving analysis reports

NEVER answer questions directly. ALWAYS delegate to the appropriate agent.

For a query like "What's the current price of AAPL?":
- DELEGATE to data_analyst (do not answer yourself)

For a query like "Create a chart for TSLA":
- DELEGATE to chart_specialist (do not answer yourself)

Your job is ONLY to route requests to specialists, not to answer them yourself."""

    # Create supervisor with all agents
    workflow = create_supervisor(
        agents,
        model=model,
        prompt=supervisor_prompt,
    )

    print("Supervisor workflow created")
    return workflow

# ============================================================================
# EXECUTION
# ============================================================================

async def run_analysis(app, query: str):
    """Run a financial analysis query."""
    print(f"\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)

    try:
        result = await app.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        })

        print("\n" + "="*80)
        print("Analysis Complete!")
        print("="*80)

        # Print final messages
        if "messages" in result:
            print("\nFinal Response:")
            for msg in result["messages"][-3:]:  # Show last 3 messages
                if hasattr(msg, 'content') and msg.content:
                    print(f"\n{msg.content[:300]}...")

        return result

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

async def interactive_mode(app):
    """Run in interactive mode."""
    print("\n" + "="*80)
    print("Financial Analyst System V2 - Interactive Mode")
    print("="*80)
    print("\nCommands:")
    print("  - Enter your query")
    print("  - Type 'examples' for example queries")
    print("  - Type 'quit' to exit")
    print("\n" + "="*80)

    while True:
        try:
            query = input("\nYour query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if query.lower() == 'examples':
                print("\nExample Queries:")
                print("  1. What's Apple's current stock price?")
                print("  2. Create a 6-month chart for Tesla")
                print("  3. Compare AAPL, MSFT, and GOOGL")
                print("  4. Get news for NVIDIA")
                print("  5. Analyze Microsoft and create a report")
                continue

            if not query:
                continue

            await run_analysis(app, query)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nERROR: {e}")

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point."""
    try:
        # Check for langgraph_supervisor
        if not HAS_SUPERVISOR:
            print("\nERROR: This version requires langgraph_supervisor")
            print("Install with: pip install langgraph-supervisor")
            print("\nOr use the original version: python financial_analyst_system.py")
            return

        # Initialize system
        model, mcp_client, tools_by_category = await initialize_system()

        # Create agents
        agents = create_specialized_agents(model, tools_by_category)

        # Build supervisor workflow
        workflow = build_supervisor_workflow(agents, model)

        # Compile
        app = workflow.compile()
        print("Workflow compiled\n")

        # Run example
        print("\n" + "="*80)
        print("RUNNING EXAMPLE QUERY")
        print("="*80)

        await run_analysis(app, "What's the current price of Apple (AAPL)?")

        # Interactive mode
        print("\n" + "="*80)
        print("Entering Interactive Mode...")
        print("="*80)

        await interactive_mode(app)

    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")

if __name__ == "__main__":
    asyncio.run(main())
