#!/bin/sh

find src -type d -name "__pycache__" -exec rm -rv {} \;
find src -type f -name "*.pyc" -exec rm -v {} \;
