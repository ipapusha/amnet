#!/bin/sh
set -e # quit if any command fails
set -v # show every command as it executes
# run example forward verification files
PYTHONPATH=.. python example_verify_forward_invariance.py
PYTHONPATH=.. python example_verify_forward_invariance2.py
