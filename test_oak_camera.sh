#!/usr/bin/env bash
set -e

if [ -d "oakenv" ]; then
  echo "Activating oakenv..."
  # shellcheck disable=SC1091
  source oakenv/bin/activate
else
  echo "oakenv was not found in the current project folder."
  echo "Create it first, for example:"
  echo "python3 -m venv oakenv"
  echo "Then install dependencies with:"
  echo "pip install -r requirements.txt"
  exit 1
fi

echo "Running OAK-D camera test..."
python3 oak_test.py
