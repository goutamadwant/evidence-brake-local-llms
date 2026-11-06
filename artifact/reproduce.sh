#!/bin/sh
# Regenerates the results manifest, generated numbers, table and figure from experiments/registry.json and the run outputs.
# Pass --registry and --frames to read them from another location.
set -e
cd "$(dirname "$0")"
python3 analyze.py "$@"
