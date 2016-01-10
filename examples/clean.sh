#!/bin/sh

rm -v *.png
rm -v *.svg

# Python 3.x

find . -type d -name "__pycache__" -exec rm -rv "{}" \;

# Python 2.x

find . -type f -name "*.pyc" -exec rm -v "{}" \;
