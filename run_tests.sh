#!/bin/sh
set -e
PYTHONPATH=. python tests/test_util.py
PYTHONPATH=. python tests/test_atoms.py
PYTHONPATH=. python tests/test_operator_overloads.py
PYTHONPATH=. python tests/test_smt.py
PYTHONPATH=. python tests/test_tree.py
PYTHONPATH=. python tests/test_smt_relu.py
PYTHONPATH=. python tests/test_lyap.py
#PYTHONPATH=. python tests/test_tf.py
