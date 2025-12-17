# Test Suite

This directory contains tests for the Financial Analyst Multi-Agent System.

## Test Files

### 1. test_system.py
**Purpose**: Component validation test
**What it tests**:
- Ollama connection and model availability
- MCP server files exist
- MCP client initialization
- Python dependencies
- Simple end-to-end query

**Run**: `python tests/test_system.py`

**Use case**: Quick validation that all components are installed and working

---

### 2. test_manual_quick.py
**Purpose**: Quick test of manual graph orchestration
**What it tests**:
- Manual graph system initialization
- Single simple query ("What's the current price of Apple?")
- Tool usage verification

**Run**: `python tests/test_manual_quick.py`

**Use case**: Fast verification that manual graph system is working

---

### 3. test_supervisor_quick.py
**Purpose**: Quick test of supervisor pattern orchestration
**What it tests**:
- Supervisor system initialization
- Single simple query ("What's the current price of Apple?")
- Agent delegation and tool usage

**Run**: `python tests/test_supervisor_quick.py`

**Use case**: Fast verification that supervisor pattern is working

---

### 4. test_comprehensive.py
**Purpose**: Full end-to-end analysis with report generation
**What it tests**:
- Complete workflow: data retrieval, chart creation, news analysis, report saving
- All 4 specialist agents (data_analyst, chart_specialist, news_analyst, report_writer)
- MCP tool integration
- File output generation

**Run**: `python tests/test_comprehensive.py`

**Use case**: Comprehensive validation that generates actual output files (charts and reports)

**Expected outputs**:
- Chart: `outputs/charts/AAPL_3mo.png`
- Report: `outputs/reports/*.md`

---

## Running All Tests

```bash
# Component tests
python tests/test_system.py

# Quick functionality tests
python tests/test_manual_quick.py
python tests/test_supervisor_quick.py

# Full end-to-end test with report generation
python tests/test_comprehensive.py
```

## Test Organization

- **Component tests**: Verify individual pieces work
- **Quick tests**: Fast smoke tests for each orchestration method
- **Comprehensive test**: Full workflow validation with file outputs

All tests are non-interactive and can be run in CI/CD pipelines.
