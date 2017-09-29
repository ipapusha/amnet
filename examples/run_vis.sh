#!/bin/sh
set -e

# run example visualization files
PYTHONPATH=.. python example_vis.py
PYTHONPATH=.. python example_vis2.py
PYTHONPATH=.. python example_vis3.py

# convert gv to png using dot
#for f in vis/*.gv; do
#	dot -Tpng $f > "$f.png"
#done

# convert pdf to png using imagemagick
for f in vis/*.gv.pdf; do
	pngout="${f%.gv.pdf}.gv.png"
	convert -density 480 $f -resize 25% $pngout
done
