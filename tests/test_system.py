"""
Simple test script to verify the system setup
Tests each component individually before running the full system
"""

import asyncio
import sys
from pathlib import Path

async def test_ollama():
    """Test Ollama connection and model availability."""
    print("\nTesting Ollama connection...")
    try:
        from langchain_ollama import ChatOllama
        model = ChatOllama(model="granite4:3b", base_url="http://localhost:11434")
        response = model.invoke("Say 'OK' if you can read this")
        print(f"   PASS: Ollama is working! Response: {response.content[:50]}")
        return True
    except Exception as e:
        print(f"   FAIL: Ollama error: {e}")
        print("   Hint: Make sure Ollama is running: ollama serve")
        print("   Hint: Pull the model: ollama pull granite4:3b")
        return False

async def test_mcp_servers():
    """Test individual MCP servers."""
    print("\nTesting MCP servers...")
    server_dir = Path(__file__).parent.parent / "mcp_servers"
    
    servers = [
        "server_stock_data.py",
        "server_plot.py", 
        "server_news.py",
        "server_report.py"
    ]
    
    all_good = True
    for server in servers:
        server_path = server_dir / server
        if server_path.exists():
            print(f"   ✅ Found {server}")
        else:
            print(f"   ❌ Missing {server}")
            all_good = False
    
    return all_good

async def test_mcp_client():
    """Test MCP client initialization."""
    print("\nTesting MCP client...")
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        server_dir = Path(__file__).parent.parent / "mcp_servers"
        
        # Test with just one server
        test_config = {
            "stock_data": {
                "transport": "stdio",
                "command": "python",
                "args": [str(server_dir / "server_stock_data.py")],
            }
        }
        
        client = MultiServerMCPClient(test_config)
        tools = await client.get_tools()

        print(f"   PASS: MCP client working! Loaded {len(tools)} tools")
        for tool in tools:
            print(f"      - {tool.name}")
        return True

    except Exception as e:
        print(f"   FAIL: MCP client error: {e}")
        return False

async def test_dependencies():
    """Test all required dependencies."""
    print("\nTesting dependencies...")

    required = [
        ("langchain", "LangChain"),
        ("langgraph", "LangGraph"),
        ("langchain_mcp_adapters", "MCP Adapters"),
        ("langchain_ollama", "Ollama Integration"),
        ("yfinance", "Yahoo Finance"),
        ("matplotlib", "Matplotlib"),
    ]

    all_good = True
    for module, name in required:
        try:
            __import__(module)
            print(f"   PASS: {name}")
        except ImportError:
            print(f"   FAIL: {name} - run: pip install -r requirements.txt")
            all_good = False

    return all_good

async def run_simple_query():
    """Run a simple end-to-end test."""
    print("\nRunning simple end-to-end test...")
    try:
        from langchain_ollama import ChatOllama
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import HumanMessage

        # Initialize
        model = ChatOllama(model="granite4:3b", temperature=0)

        server_dir = Path(__file__).parent.parent / "mcp_servers"
        mcp_client = MultiServerMCPClient({
            "stock_data": {
                "transport": "stdio",
                "command": "python",
                "args": [str(server_dir / "server_stock_data.py")],
            }
        })
        
        tools = await mcp_client.get_tools()
        agent = create_react_agent(model, tools)
        
        # Test query
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="Use the 'get_stock_price' tool to check the price of AAPL. You MUST use the tool.")]
        })

        print(f"   PASS: End-to-end test passed!")
        print(f"   Response: {result['messages'][-1].content[:100]}")
        return True

    except Exception as e:
        print(f"   FAIL: End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("="*70)
    print("Financial Analyst System - Component Tests")
    print("="*70)

    results = {
        "Dependencies": await test_dependencies(),
        "Ollama": await test_ollama(),
        "MCP Servers": await test_mcp_servers(),
        "MCP Client": await test_mcp_client(),
        "End-to-End": await run_simple_query(),
    }

    print("\n" + "="*70)
    print("Test Results Summary")
    print("="*70)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:.<50} {status}")

    all_passed = all(results.values())

    print("\n" + "="*70)
    if all_passed:
        print("All tests passed! System is ready to run.")
        print("Run the main system with: python financial_analyst_system_supervisor.py")
    else:
        print("Some tests failed. Please fix the issues above.")
        print("Check README.md for troubleshooting guide.")
    print("="*70)

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
