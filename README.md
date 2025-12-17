# Multi-Agent Financial Analyst System

> A Practice Laboratory for Building Multi-Agent Systems with MCP

This repository demonstrates how to build production-ready multi-agent systems using **LangGraph**, **Model Context Protocol (MCP)**, and **local LLMs**. It showcases two different orchestration patterns for coordinating specialized AI agents to perform complex financial analysis tasks.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Two Orchestration Methods](#two-orchestration-methods)
4. [Repository Structure](#repository-structure)
5. [Quick Start](#quick-start)
6. [Understanding MCP](#understanding-mcp-model-context-protocol)
7. [How the Agents Work](#how-the-agents-work)
8. [Orchestration Deep Dive](#orchestration-deep-dive)
9. [Example Outputs](#example-outputs)
10. [Testing & Comparison](#testing--comparison)
11. [Troubleshooting](#troubleshooting)


## Repository Structure

```
multi_agent/
│
├── Core Orchestration Files (2)
│   ├── financial_analyst_system_manual_graph.py    # Method 1: Explicit control
│   └── financial_analyst_system_supervisor.py      # Method 2: Automatic routing
│
├── MCP Server Implementations (4)
│   └── mcp_servers/
│       ├── server_stock_data.py    # Yahoo Finance integration (3 tools)
│       ├── server_plot.py          # Chart generation (2 tools)
│       ├── server_news.py          # News retrieval (1 tool)
│       └── server_report.py        # Report saving (1 tool)
│
├── Testing & Scripts
│   ├── tests/
│   │   ├── README.md                 # Test documentation
│   │   ├── test_system.py            # Component validation
│   │   ├── test_manual_quick.py      # Manual graph quick test
│   │   ├── test_supervisor_quick.py  # Supervisor quick test
│   │   └── test_comprehensive.py     # Full workflow + report generation
│   └── scripts/
│       ├── compare_orchestrations.py # Side-by-side comparison
│       ├── demo_full_workflow.py     # Full pipeline demo
│       └── run_complex_test.py       # Complex test scenarios
│
├── Generated Outputs (auto-created)
│   └── outputs/
│       ├── charts/                 # PNG visualizations
│       └── reports/                # Markdown analysis reports
│
├── Configuration Files
│   ├── requirements.txt           # Python dependencies
│   ├── setup_conda_uv.sh          # Automated setup script
│   └── .gitignore                 # Git ignore rules
│
└── Documentation
    └── README.md          
```


## System Overview

### What Does This System Do?

The **Financial Analyst Multi-Agent System** analyzes stocks by:

1. **Data Collection**: Fetches real-time stock prices, historical data, and company information
2. **Visualization**: Creates price charts and comparisons
3. **News Analysis**: Retrieves and analyzes market news with sentiment scoring
4. **Report Generation**: Produces markdown reports with all findings

### Key Features

- **4 Specialized Agents**: Data Analyst, Chart Specialist, News Analyst, Report Writer
- **4 MCP Servers and 7 Tools**: Stock data, charts, news, report generation
- **2 Orchestration Patterns**: Manual graph construction vs. automatic supervision
- **Local LLM**: Runs entirely on Ollama (no cloud APIs needed)
- **Production Ready**: Proper error handling, state management, checkpointing


## Architecture Components

### 1. LLM Backend: Ollama

**Model**: `granite4:3b` (IBM's open-source model)

```python
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="granite4:3b",
    base_url="http://localhost:11434"
)
```

### 2. MCP

The defined tools run as independent server processes communicating via STDIO.

**Our MCP Servers**:
```
mcp_servers/
├── server_stock_data.py    # Yahoo Finance API integration
├── server_plot.py          # Matplotlib chart generation
├── server_news.py          # Financial news retrieval
└── server_report.py        # Markdown report saving
```

Each server exposes tools that agents can invoke:

```python
# Example: Stock Data Server
mcp = FastMCP("StockData")

@mcp.tool()
def get_stock_price(ticker: str) -> dict:
    """Get current stock price and trading data."""
    stock = yf.Ticker(ticker)
    return {
        "current_price": price,
        "day_high": high,
        "volume": volume,
        # ... more data
    }

mcp.run(transport="stdio")
```

### 3. LangGraph: Workflow Engine

`LangGraph` is a framework -- derived from LangChain -- for building stateful, multi-actor applications with LLMs.

**Purpose**: Orchestrates agent interactions, manages state, handles routing
**Key Concepts**:
- **Nodes**: Individual agents or functions
- **Edges**: Connections between nodes
- **State**: Shared data structure passed between nodes
- **Graph**: The complete workflow definition

### 4. Specialized Agents

Each agent is a **ReAct (Reasoning + Acting)** agent with:
- Specific tools for its domain
- Custom system prompt defining its role
- Access to conversation history
- Ability to invoke tools and reason about results

**Agent Roster**:

| Agent | Tools | Purpose |
|-------|-------|---------|
| **Supervisor** | None | Routes queries to appropriate specialists |
| **Data Analyst** | `get_stock_price`, `get_historical_data`, `get_stock_info` | Retrieves and analyzes financial data |
| **Chart Specialist** | `create_chart`, `create_comparison` | Creates visualizations |
| **News Analyst** | `get_stock_news` | Fetches news and performs sentiment analysis |
| **Report Writer** | `save_report` | Compiles findings into markdown reports |

---

## Two Orchestration Methods

This repository provides implementations of **two different approaches** to multi-agent orchestration, each with distinct trade-offs.

### Method 1: Manual Graph Construction

**File**: [`financial_analyst_system_manual_graph.py`](financial_analyst_system_manual_graph.py)

**Philosophy**: Explicit control over every aspect of agent coordination.

#### How It Works

1. **Custom State Definition**
```python
class FinancialAnalystState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next: str  # Which agent to route to
    analysis_context: dict  # Shared data across agents
```

2. **Supervisor Node with Routing Logic**
```python
async def supervisor_node(state: FinancialAnalystState) -> dict:
    """Decides which agent should handle the request."""
    messages = state["messages"]

    # Supervisor analyzes conversation and decides routing
    routing_messages = [SystemMessage(content=SUPERVISOR_PROMPT), *messages]
    response = await model.ainvoke(routing_messages)

    # Parse decision from response
    decision = response.content.strip().lower()
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

    return {
        "next": next_step,
        "messages": [AIMessage(content=f"Routing to {decision}", name="supervisor")]
    }
```

3. **Explicit Graph Construction**
```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(FinancialAnalystState)

# Add all nodes
builder.add_node("supervisor", supervisor_node)
builder.add_node("data_analyst", create_agent_node(data_analyst, "data_analyst"))
builder.add_node("chart_specialist", create_agent_node(chart_specialist, "chart_specialist"))
builder.add_node("news_analyst", create_agent_node(news_analyst, "news_analyst"))
builder.add_node("report_writer", create_agent_node(report_writer, "report_writer"))

# Define routing edges
builder.add_edge(START, "supervisor")

# Conditional routing from supervisor
builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,  # Function that reads state["next"]
    {
        "data_analyst": "data_analyst",
        "chart_specialist": "chart_specialist",
        "news_analyst": "news_analyst",
        "report_writer": "report_writer",
        END: END
    }
)

# All agents return to supervisor
for agent in ["data_analyst", "chart_specialist", "news_analyst", "report_writer"]:
    builder.add_edge(agent, "supervisor")

# Compile the graph
app = builder.compile(checkpointer=MemorySaver())
```

#### Advantages
- **Full Control**: Every routing decision is explicit and customizable
- **Custom State**: Can add any fields to state for data sharing
- **Detailed Prompts**: Complete control over supervisor reasoning
- **Debugging**: Clear visibility into workflow execution
- **Flexibility**: Easy to add custom logic or routing rules


---

### Method 2: Supervisor Pattern (Automatic)

**File**: [`financial_analyst_system_supervisor.py`](financial_analyst_system_supervisor.py)

**Philosophy**: Let the framework handle orchestration automatically.

#### How It Works

1. **Create Specialized Agents**
```python
data_analyst = create_react_agent(
    model=model,
    tools=tools_by_category["data"],
    prompt=DATA_ANALYST_PROMPT, # < --- Custom system prompt
    name="data_analyst"  # Name is crucial for routing
)

chart_specialist = create_react_agent(
    model=model,
    tools=tools_by_category["chart"],
    prompt=CHART_SPECIALIST_PROMPT,
    name="chart_specialist"
)

# ... more agents
```

2. **Single Function Call for Supervision**
```python
from langgraph_supervisor import create_supervisor

# Build supervisor workflow
workflow = create_supervisor(
    agents,  # List of all specialized agents
    model=model,
)

app = workflow.compile()
```

That's it! The supervisor pattern automatically:
- Creates a supervisor agent
- Sets up routing logic
- Manages state transitions
- Handles agent coordination

#### Advantages
- **Simplicity**: 75% less boilerplate code
- **Fast Development**: Rapid prototyping and iteration
- **Automatic Routing**: LLM decides agent delegation
- **Maintainability**: Less code to maintain
- **Built-in Best Practices**: Framework handles edge cases


---

## Quick Start - Installation

```bash
bash setup_conda_uv.sh

# This script will:
# - Check/start Ollama
# - Pull the granite4:3b model
# - Create conda environment
# - Install all dependencies
# - Run system tests
```

### Running the System

#### Method 1: Manual Graph
```bash
conda activate financial_analyst_env
python financial_analyst_system_manual_graph.py
```

**Interactive prompt appears:**
```
Your query: Analyze Apple stock and create a report with charts
```

#### Method 2: Supervisor Pattern
```bash
conda activate financial_analyst_env
python financial_analyst_system_supervisor.py
```

#### Compare Both Methods
```bash
python scripts/compare_orchestrations.py "What's Tesla's stock price?"

# Or interactive mode
python scripts/compare_orchestrations.py -i
```



### Acknowledgements

This repository is based on the [LangGraph](https://github.com/langchain-ai/langgraph) library. 

### Author
Lorenzo Molfetta

Email: lorenzo.molfetta@unibo.it

[GitHub](https://github.com/LorMolf)
