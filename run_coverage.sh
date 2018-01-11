#!/bin/sh
# pip install coverage
set -e
coverage erase
PYTHONPATH=. coverage run -a --source=. tests/test_atoms.py
PYTHONPATH=. coverage run -a --source=. tests/test_operator_overloads.py
PYTHONPATH=. coverage run -a --source=. tests/test_smt.py
PYTHONPATH=. coverage run -a --source=. tests/test_tree.py
PYTHONPATH=. coverage run -a --source=. tests/test_tf.py
coverage report -m
coverage html

# consider also: (from https://stackoverflow.com/q/34728734)
#python -m unittest discover -s tests/
#coverage run -m unittest discover -s tests/
#coverage report -m

