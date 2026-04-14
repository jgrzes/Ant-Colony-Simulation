#!/bin/bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "------------------------------------------------"
echo "Setup complete! To start, run:"
echo "source .venv/bin/activate"
echo "------------------------------------------------"
