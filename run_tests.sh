#!/bin/sh

# DOCTESTS ####################################################################

echo
echo
python3 -m doctest ./ailib/optimize/functions/unconstrained.py
if [ $? -ne 0 ]; then
    exit 1
fi

# UNITTESTS ###################################################################

pytest
