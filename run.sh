#!/bin/bash
# Launcher: ./run.sh <record|detect|enroll|recognize|forget> [args]
cd "$(dirname "$0")"
case "$1" in
  record|detect|enroll|recognize|forget) script="$1.py"; shift; exec .venv/bin/python -W ignore "$script" "$@" ;;
  *) echo "Usage: ./run.sh record | detect <photo> | enroll <name> | recognize | forget [name]"; exit 1 ;;
esac
