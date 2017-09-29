#!/bin/sh
set -e

# run example visualization files
PYTHONPATH=.. python example_vis.py
PYTHONPATH=.. python example_vis2.py
PYTHONPATH=.. python example_vis3.py

# convert gv to png
for f in vis/*.gv; do
	dot -Tpng $f > "$f.png"
done
