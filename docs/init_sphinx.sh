#!/bin/sh

PROJECT_VERSION=$(python -c "import sys ; sys.path.append('..') ; print('.'.join(__import__('TODO_PYTHON_PACKAGE_NAME').__version__.split('.')[:2]))")
PROJECT_RELEASE=$(python -c "import sys ; sys.path.append('..') ; print('.'.join(__import__('TODO_PYTHON_PACKAGE_NAME').__version__.split('.')))")

sphinx-quickstart \
    --sep \
    --project="TODO_PROJECT_NAME" \
    --author="TODO_AUTHOR_NAME" \
    -v "${PROJECT_VERSION}" \             # The short X.Y version.
    --release="${PROJECT_RELEASE}" \      # The full version, including alpha/beta/rc tags.
    --language=en \
    --suffix=".rst" \
    --master="index" \
    --ext-autodoc \
    --ext-doctest \
    --ext-intersphinx \
    --ext-todo \
    --ext-coverage \
    #--ext-imgmath \
    --ext-mathjax \
    #--ext-ifconfig \
    --ext-viewcode \
    #--ext-githubpages \
    --makefile \
    --batchfile \


