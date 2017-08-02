#!/bin/sh

# The MIT License
#
# Copyright (c) 2016 Jérémie DECOCK <jd.jdhp@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


# SAFETY TESTS ################################################################

# TODO: assert "$(dirname "$0") == "$(pwd)"
#       ...


# LOAD VARIABLES ##############################################################

. ./meta.sh

echo "AUTHOR_NAME: ${AUTHOR_NAME}"
echo "AUTHOR_EMAIL: ${AUTHOR_EMAIL}"
echo "AUTHOR_WEB_SITE: ${AUTHOR_WEB_SITE}"
echo "COPYRIGHT_NOTICE: ${COPYRIGHT_NOTICE}"
echo "PROJECT_INITIAL_DATE: ${PROJECT_INITIAL_DATE}"
echo "PROJECT_SHORT_DESC: ${PROJECT_SHORT_DESC}"
echo "PROJECT_NAME: ${PROJECT_NAME}"
echo "PYTHON_PACKAGE_NAME: ${PYTHON_PACKAGE_NAME}"
echo "PROJECT_GITHUB_ACCOUNT: ${PROJECT_GITHUB_ACCOUNT}"
echo "PROJECT_GITHUB_REPOSITORY_NAME: ${PROJECT_GITHUB_REPOSITORY_NAME}"
echo "PROJECT_GITHUB_URL ${PROJECT_GITHUB_URL}"
echo "PROJECT_ISSUE_TRACKER_URL: ${PROJECT_ISSUE_TRACKER_URL}"
echo "PROJECT_PYPI_URL: ${PROJECT_PYPI_URL}"
echo "PROJECT_WEB_SITE_URL: ${PROJECT_WEB_SITE_URL}"
echo "PROJECT_ONLINE_DOCUMENTATION_URL: ${PROJECT_ONLINE_DOCUMENTATION_URL}"
echo "PROJECT_ONLINE_API_DOCUMENTATION_URL: ${PROJECT_ONLINE_API_DOCUMENTATION_URL}"


# SETUP GIT ###################################################################

git remote rename origin skeleton
git remote add origin git@github.com:${PROJECT_GITHUB_ACCOUNT}/${PROJECT_GITHUB_REPOSITORY_NAME}.git
git push -u origin master

git submodule init
git submodule update


# MAKE SUBSTITUTIONS ##########################################################

sed -i "" \
    -e "s/TODO_AUTHOR_NAME/${AUTHOR_NAME}/g" \
    -e "s/TODO_AUTHOR_EMAIL/${AUTHOR_EMAIL}/g" \
    -e "s TODO_AUTHOR_WEB_SITE ${AUTHOR_WEB_SITE} g" \
    -e "s|TODO_COPYRIGHT_NOTICE|${COPYRIGHT_NOTICE}|g" \
    -e "s|TODO_PROJECT_INITIAL_DATE|${PROJECT_INITIAL_DATE}|g" \
    -e "s|TODO_PROJECT_SHORT_DESC|${PROJECT_SHORT_DESC}|g" \
    -e "s;TODO_PROJECT_NAME;${PROJECT_NAME};g" \
    -e "s/TODO_PYTHON_PACKAGE_NAME/${PYTHON_PACKAGE_NAME}/g" \
    -e "s/TODO_PROJECT_GITHUB_ACCOUNT/${PROJECT_GITHUB_ACCOUNT}/g" \
    -e "s/TODO_PROJECT_GITHUB_REPOSITORY_NAME/${PROJECT_GITHUB_REPOSITORY_NAME}/g" \
    -e "s TODO_PROJECT_GITHUB_URL ${PROJECT_GITHUB_URL} g" \
    -e "s TODO_PROJECT_ISSUE_TRACKER_URL ${PROJECT_ISSUE_TRACKER_URL} g" \
    -e "s TODO_PROJECT_PYPI_URL ${PROJECT_PYPI_URL} g" \
    -e "s TODO_PROJECT_WEB_SITE_URL ${PROJECT_WEB_SITE_URL} g" \
    -e "s TODO_PROJECT_ONLINE_DOCUMENTATION_URL ${PROJECT_ONLINE_DOCUMENTATION_URL} g" \
    -e "s TODO_PROJECT_ONLINE_API_DOCUMENTATION_URL ${PROJECT_ONLINE_API_DOCUMENTATION_URL} g" \
    AUTHORS \
    CHANGES.rst \
    LICENSE \
    meta.make \
    README.rst \
    setup.py \
    docs/api.rst \
    docs/conf.py \
    docs/developer.rst \
    docs/index.rst \
    docs/init_sphinx.sh \
    docs/intro.rst \
    docs/make.bat \
    docs/Makefile \
    TODO_PYTHON_PACKAGE_NAME/__init__.py


# FIX TITLES UNDERLINE LENGTH IN RESTRUCTUREDTEXT FILES #######################

PROJECT_NAME_UNDERLINE=$(echo "${PROJECT_NAME}" | tr '[:print:]' '=')

sed -i "" \
    -e "s/^====$/${PROJECT_NAME_UNDERLINE}/" \
    README.rst

sed -i "" \
    -e "s/^====$/${PROJECT_NAME_UNDERLINE}/" \
    docs/index.rst


# RENAME THE ROOT PACKAGE DIRECTORY ###########################################

mv -v TODO_PYTHON_PACKAGE_NAME "${PYTHON_PACKAGE_NAME}"

