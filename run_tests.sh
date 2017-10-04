#!/bin/sh
set -e
PYTHONPATH=. python tests/test_atoms.py
PYTHONPATH=. python tests/test_smt.py
#PYTHONPATH=. python tests/test_lyap.py
#PYTHONPATH=. python tests/test_tf.py
