#!/bin/bash

# Start the Flask application with Gunicorn
echo "🚀 Starting TechnoColabs Delivery AI on Render..."
echo "📊 Loading ML model components..."

# Set environment variables
export PYTHONPATH=/app
export PORT=${PORT:-5000}

# Pre-load the model to ensure it's available
echo "🔧 Pre-loading ML model..."
python -c "from api.index import load_model_components; result = load_model_components(); print(f'Model load result: {result}')"

# Start the application
exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --keep-alive 5 api.index:app
