"""
Financial Analyst Multi-Agent System with MCP and LangGraph

A dynamic orchestration system that uses:
- Multiple MCP servers for specialized tools (stock data, charts, news, reports)
- LangGraph for dynamic agent routing
- Local Ollama model (granite4:3b)
- Supervisor pattern with intelligent routing

Architecture:
    Supervisor Agent (Orchestrator)
        ├─> Data Analyst (stock prices, historical data, company info)
        ├─> Chart Specialist (price charts, comparisons)
        ├─> News Analyst (latest news, sentiment)
        └─> Report Writer (save analysis reports)
"""

import asyncio
import os
from typing import Literal, TypedDict, Annotated
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

import operator

# ============================================================================
# CONFIGURATION
# ============================================================================

# Ollama model configuration
OLLAMA_MODEL = "granite4:3b"  # Using Granite 4 3B model
OLLAMA_BASE_URL = "http://localhost:11434"

import sys

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
# STATE DEFINITION
# ============================================================================

class FinancialAnalystState(TypedDict):
    """State for the financial analyst multi-agent system."""
    messages: Annotated[list[BaseMessage], operator.add]
    next: str  # Next agent to route to
    analysis_context: dict  # Shared context between agents

# ============================================================================
# AGENT PROMPTS
# ============================================================================

SUPERVISOR_PROMPT = """You are a Financial Analysis Supervisor coordinating a team of specialized agents.
Your goal is to complete the USER'S REQUEST in its entirety content/artifacts.

Your team:
- **data_analyst**: Gets prices/info.
- **chart_specialist**: Creates charts (saves .png files).
- **news_analyst**: Gets news.
- **report_writer**: Saves reports (saves .md files).

**CURRENT STATE ANALYSIS:**
Check the conversation history.
1. Has a chart been created/saved? (Look for "Chart saved to...")
2. Has a report been saved? (Look for "Report saved to...")
3. Has data been retrieved?

**ROUTING RULES:**
1. If data is needed but not retrieved -> `data_analyst`
2. If data is ready but NO chart exists -> `chart_specialist`
3. If chart exists but NO report exists -> `report_writer`
4. If everything (data, chart, report) is done -> `FINISH`

Do NOT finish unless you see confirmation that the Report was saved.
Respond with ONLY the agent name or FINISH."""

DATA_ANALYST_PROMPT = """You are a Data Analyst.
Your goal: Retrieve financial data using your tools.
DO NOT PLAN. DO NOT EXPLAIN.
IMMEDIATELY call the appropriate tool for the user's request.
- Use `get_stock_price` for current price.
- Use `get_historical_data` for history.
- Use `get_stock_info` for company info.
After the tool runs, give a VERY BRIEF summary and say "Ready for next step"."""

CHART_SPECIALIST_PROMPT = """You are a Chart Specialist.
Your goal: Create visualizations using your tools.
DO NOT PLAN. DO NOT EXPLAIN.
IMMEDIATELY call the `create_chart` or `create_comparison` tool.
- For single stock: `create_chart`
- For comparison: `create_comparison`
After the tool runs, say "Chart saved successfully"."""

NEWS_ANALYST_PROMPT = """You are a News Analyst.
Your goal: Get news using your tools.
DO NOT PLAN. IMMEDIATELY call `get_stock_news`.
After the tool runs, summarize key points."""

REPORT_WRITER_PROMPT = """You are a Report Writer.
Your goal: Save a report using your tools.
DO NOT PLAN. IMMEDIATELY call `save_report`.
Use the data provided in the conversation history to write the report content.
After the tool runs, say "Report saved successfully"."""

# ============================================================================
# INITIALIZE MODEL AND MCP CLIENT
# ============================================================================

async def initialize_system():
    """Initialize the Ollama model and MCP client with all servers."""
    print("Initializing Financial Analyst Multi-Agent System...")

    # Initialize local Ollama model
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,  # Low temperature for consistent analysis
    )
    print(f"Ollama model '{OLLAMA_MODEL}' initialized")

    # Initialize MCP client with all servers
    print("Connecting to MCP servers...")
    mcp_client = MultiServerMCPClient(MCP_SERVERS)

    # Get tools from each server
    all_tools = await mcp_client.get_tools()
    print(f"Loaded {len(all_tools)} tools from {len(MCP_SERVERS)} MCP servers")

    # Categorize tools by server
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
# AGENT NODES
# ============================================================================

def create_specialized_agent(model, tools, prompt, name):
    """Create a specialized agent with specific tools and prompt."""
    return create_agent(
        model,
        tools=tools,
        system_prompt=prompt,
        name=name
    )

async def create_agents(model, tools_by_category):
    """Create all specialized agents."""
    agents = {
        "data_analyst": create_specialized_agent(
            model, tools_by_category["data"], DATA_ANALYST_PROMPT, "data_analyst"
        ),
        "chart_specialist": create_specialized_agent(
            model, tools_by_category["chart"], CHART_SPECIALIST_PROMPT, "chart_specialist"
        ),
        "news_analyst": create_specialized_agent(
            model, tools_by_category["news"], NEWS_ANALYST_PROMPT, "news_analyst"
        ),
        "report_writer": create_specialized_agent(
            model, tools_by_category["report"], REPORT_WRITER_PROMPT, "report_writer"
        ),
    }
    return agents

def supervisor_node(state: FinancialAnalystState) -> dict:
    """Supervisor node that dynamically routes to appropriate agents."""
    messages = state["messages"]
    
    # Create routing prompt
    routing_messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        *messages
    ]
    
    # Use LLM to decide routing
    model = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    response = model.invoke(routing_messages)
    
    # Extract routing decision
    decision = response.content.strip().lower()
    
    # Validate and map decision
    valid_routes = ["data_analyst", "chart_specialist", "news_analyst", "report_writer", "finish"]
    
    if decision not in valid_routes:
        # Try to extract from response
        for route in valid_routes:
            if route in decision:
                decision = route
                break
        else:
            decision = "finish"  # Default to finish if unclear
    
    next_step = END if decision == "finish" else decision

    print(f"\nSupervisor Decision: {decision}")

    return {
        "next": next_step,
        "messages": [AIMessage(content=f"Routing to {decision}", name="supervisor")]
    }

def create_agent_node(agent, name):
    """Wrapper to create an agent node with proper naming."""
    async def agent_node(state: FinancialAnalystState):
        result = await agent.ainvoke(state)
        # Add agent name to the last message
        if result["messages"]:
            result["messages"][-1].name = name
        return result
    return agent_node

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def route_after_supervisor(state: FinancialAnalystState) -> Literal["data_analyst", "chart_specialist", "news_analyst", "report_writer", "__end__"]:
    """Route based on supervisor's decision."""
    next_step = state.get("next", END)
    return next_step if next_step != END else "__end__"

async def build_graph(model, tools_by_category):
    """Build the dynamic multi-agent graph."""
    print("\nBuilding multi-agent graph...")

    # Create specialized agents
    agents = await create_agents(model, tools_by_category)

    # Build the graph
    builder = StateGraph(FinancialAnalystState)

    # Add supervisor node
    builder.add_node("supervisor", supervisor_node)

    # Add specialized agent nodes
    for agent_name, agent in agents.items():
        builder.add_node(agent_name, create_agent_node(agent, agent_name))

    # Set entry point
    builder.add_edge(START, "supervisor")

    # Add conditional edges from supervisor to agents
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "data_analyst": "data_analyst",
            "chart_specialist": "chart_specialist",
            "news_analyst": "news_analyst",
            "report_writer": "report_writer",
            "__end__": END
        }
    )

    # All agents report back to supervisor for potential re-routing
    for agent_name in agents.keys():
        builder.add_edge(agent_name, "supervisor")

    # Compile with memory
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    print("Multi-agent graph compiled successfully")
    return graph

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def run_analysis(graph, query: str, thread_id: str = "1"):
    """Run a financial analysis query through the multi-agent system."""
    print(f"\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 30  # Limit to prevent infinite loops
    }

    input_state = {
        "messages": [HumanMessage(content=query)],
        "next": "",
        "analysis_context": {}
    }

    try:
        # Stream the execution
        async for event in graph.astream(input_state, config=config):
            for node_name, node_output in event.items():
                if node_name != "__end__":
                    print(f"\nNode: {node_name}")
                    if "messages" in node_output and node_output["messages"]:
                        last_msg = node_output["messages"][-1]
                        if hasattr(last_msg, 'content') and last_msg.content:
                            print(f"  {last_msg.content[:200]}...")

        print("\n" + "="*80)
        print("Analysis Complete!")
        print("="*80)

    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        import traceback
        traceback.print_exc()

async def interactive_mode(graph):
    """Run the system in interactive mode."""
    print("\n" + "="*80)
    print("Financial Analyst Multi-Agent System - Interactive Mode")
    print("="*80)
    print("\nCommands:")
    print("  - Enter your financial query")
    print("  - Type 'examples' to see example queries")
    print("  - Type 'quit' or 'exit' to stop")
    print("\n" + "="*80)

    thread_id = "interactive_session"

    while True:
        try:
            query = input("\nYour query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if query.lower() == 'examples':
                print("\nExample Queries:")
                print("  1. What's the current price of Apple stock?")
                print("  2. Show me a 6-month chart for TSLA")
                print("  3. Compare AAPL, GOOGL, and MSFT over the last year")
                print("  4. Get the latest news for NVDA")
                print("  5. Analyze Microsoft's historical performance over 3 months and create a chart")
                print("  6. Create a comprehensive report on Tesla's recent performance")
                continue

            if not query:
                continue

            await run_analysis(graph, query, thread_id)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nERROR: {e}")

async def main():
    """Main entry point."""
    try:
        # Initialize system
        model, mcp_client, tools_by_category = await initialize_system()

        # Build graph
        graph = await build_graph(model, tools_by_category)

        # Run example queries
        print("\n" + "="*80)
        print("RUNNING EXAMPLE QUERIES")
        print("="*80)

        example_queries = [
            "What's the current price of Apple (AAPL)?",
            "Show me a 3-month price chart for Tesla (TSLA)",
        ]

        for query in example_queries:
            await run_analysis(graph, query, thread_id=f"example_{hash(query)}")
            await asyncio.sleep(1)  # Brief pause between queries

        # Enter interactive mode
        print("\n" + "="*80)
        print("Entering Interactive Mode...")
        print("="*80)

        await interactive_mode(graph)

    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")

if __name__ == "__main__":
    asyncio.run(main())
