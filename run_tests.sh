#!/bin/sh
set -e
PYTHONPATH=. python tests/test_atoms.py
PYTHONPATH=. python tests/test_smt.py
