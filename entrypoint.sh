#!/bin/sh
# This script runs the Chainlit app, binding to Render's dynamic PORT (falls back to 8000 locally)

exec chainlit run app.py --host 0.0.0.0 --port ${PORT:-8000} --headless