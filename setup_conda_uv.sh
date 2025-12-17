#!/bin/bash
set -e

ENV_NAME="financial_analyst_env"
PYTHON_VERSION="3.11"

echo "================================================================"
echo "Financial Analyst System - Setup"
echo "================================================================"

# Check/Start Ollama
echo "Checking Ollama service..."
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama not found. Please install it first."
    exit 1
fi

if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Starting Ollama in background..."
    ollama serve &
    sleep 5
fi
echo "Ollama is running"

# Pull Model
echo ""
echo "Pulling granite4:3b model..."
ollama pull granite4:3b

# Create Conda Environment
echo ""
echo "Setting up conda environment: $ENV_NAME..."
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists. Skipping creation."
else
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

# Install dependencies with uv
echo ""
echo "Installing dependencies..."
conda run -n $ENV_NAME pip install uv
conda run -n $ENV_NAME uv pip install -r requirements.txt

# Verify structure
echo ""
echo "Verifying system structure..."
if [ ! -d "mcp_servers" ]; then
    echo "ERROR: mcp_servers directory missing!"
    exit 1
fi

# Run Tests
echo ""
echo "Running system tests..."
conda run -n $ENV_NAME python tests/test_system.py

echo ""
echo "================================================================"
echo "Setup complete!"
echo "================================================================"
echo "To run the system:"
echo "  conda activate $ENV_NAME"
echo "  python financial_analyst_system_manual_graph.py"
echo ""
echo "Or use the supervisor pattern:"
echo "  python financial_analyst_system_supervisor.py"
