#!/bin/bash

echo "🚀 Starting Airline Customer Service API Server"
echo "=============================================="

# Check if FastAPI is installed
python -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ FastAPI and Uvicorn are available"
else
    echo "⚠️  FastAPI not found. Installing..."
    pip install fastapi uvicorn
fi

echo ""
echo "Starting server..."
echo "📍 Server will be available at: http://localhost:8000"
echo "📚 API documentation at: http://localhost:8000/docs"
echo "🏥 Health check at: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python simple_server.py