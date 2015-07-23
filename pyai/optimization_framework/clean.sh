#!/bin/sh

rm -v *.png

# Python 3.x

#find src -type d -name "__pycache__" -exec rm -rv {} \;
rm -rvf function/__pycache__
rm -rvf optimizer/__pycache__

# Python 2.x

find . -type f -name "*.pyc" -exec rm -v {} \;
