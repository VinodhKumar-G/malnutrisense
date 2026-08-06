#!/bin/bash
# scripts/start_services.sh — Start FastAPI + Streamlit with one command
# Usage: bash scripts/start_services.sh
 
echo '======================================='
echo 'Starting MalnutriSense Services'
echo '======================================='
 
# Start FastAPI in background
echo 'Starting FastAPI on port 8000...'
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
sleep 3  # wait for model to load
 
# Check API is healthy
if curl -s http://localhost:8000/health | grep -q 'healthy'; then
    echo 'FastAPI: OK (http://localhost:8000)'
else
    echo 'FastAPI failed to start — check logs above'
    kill $API_PID
    exit 1
fi
 
# Start Streamlit in foreground
echo 'Starting Streamlit on port 8501...'
echo 'Dashboard: http://localhost:8501'
streamlit run dashboard/app.py --server.port 8501
 
# Cleanup on exit
kill $API_PID
