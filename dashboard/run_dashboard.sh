#!/bin/bash
# Run the Google-Go Analysis Dashboard

echo "Starting Google-Go Analysis Dashboard..."
echo "The dashboard will be available at http://localhost:8050"
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
python app.py
