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

source ./meta.sh

echo "TODO_AUTHOR_NAME: ${TODO_AUTHOR_NAME}"
echo "TODO_AUTHOR_EMAIL: ${TODO_AUTHOR_EMAIL}"
echo "TODO_AUTHOR_WEB_SITE: ${TODO_AUTHOR_WEB_SITE}"
echo "TODO_COPYRIGHT_NOTICE: ${TODO_COPYRIGHT_NOTICE}"
echo "TODO_PROJECT_INITIAL_DATE: ${TODO_PROJECT_INITIAL_DATE}"
echo "TODO_PROJECT_SHORT_DESC: ${TODO_PROJECT_SHORT_DESC}"
echo "TODO_PROJECT_NAME: ${TODO_PROJECT_NAME}"
echo "TODO_PROJECT_FILE_BASE_NAME: ${TODO_PROJECT_FILE_BASE_NAME}"
echo "TODO_PROJECT_GITHUB_ACCOUNT: ${TODO_PROJECT_GITHUB_ACCOUNT}"
echo "TODO_PROJECT_GITHUB_REPOSITORY_NAME: ${TODO_PROJECT_GITHUB_REPOSITORY_NAME}"
echo "TODO_PROJECT_GITHUB_URL ${TODO_PROJECT_GITHUB_URL}"
echo "TODO_PROJECT_ISSUE_TRACKER_URL: ${TODO_PROJECT_ISSUE_TRACKER_URL}"
echo "TODO_PROJECT_PYPI_URL: ${TODO_PROJECT_PYPI_URL}"
echo "TODO_PROJECT_WEB_SITE_URL: ${TODO_PROJECT_WEB_SITE_URL}"
echo "TODO_PROJECT_DOCUMENTATION_URL: ${TODO_PROJECT_DOCUMENTATION_URL}"
echo "TODO_PROJECT_API_DOCUMENTATION_URL: ${TODO_PROJECT_API_DOCUMENTATION_URL}"


# SETUP GIT ###################################################################

git remote rename origin skeleton
git remote add origin git@github.com:${TODO_PROJECT_GITHUB_ACCOUNT}/${TODO_PROJECT_GITHUB_REPOSITORY_NAME}.git
git push -u origin english-version

git submodule init
git submodule update


# MAKE SUBSTITUTIONS ##########################################################

sed -i "" \
    -e "s/TODO_AUTHOR_NAME/${TODO_AUTHOR_NAME}/g" \
    -e "s/TODO_AUTHOR_EMAIL/${TODO_AUTHOR_EMAIL}/g" \
    -e "s TODO_AUTHOR_WEB_SITE ${TODO_AUTHOR_WEB_SITE} g" \
    -e "s|TODO_COPYRIGHT_NOTICE|${TODO_COPYRIGHT_NOTICE}|g" \
    -e "s|TODO_PROJECT_INITIAL_DATE|${TODO_PROJECT_INITIAL_DATE}|g" \
    -e "s|TODO_PROJECT_SHORT_DESC|${TODO_PROJECT_SHORT_DESC}|g" \
    -e "s;TODO_PROJECT_NAME;${TODO_PROJECT_NAME};g" \
    -e "s/TODO_PROJECT_FILE_BASE_NAME/${TODO_PROJECT_FILE_BASE_NAME}/g" \
    -e "s/TODO_PROJECT_GITHUB_ACCOUNT/${TODO_PROJECT_GITHUB_ACCOUNT}/g" \
    -e "s/TODO_PROJECT_GITHUB_REPOSITORY_NAME/${TODO_PROJECT_GITHUB_REPOSITORY_NAME}/g" \
    -e "s TODO_PROJECT_GITHUB_URL ${TODO_PROJECT_GITHUB_URL} g" \
    -e "s TODO_PROJECT_ISSUE_TRACKER_URL ${TODO_PROJECT_ISSUE_TRACKER_URL} g" \
    -e "s TODO_PROJECT_PYPI_URL ${TODO_PROJECT_PYPI_URL} g" \
    -e "s TODO_PROJECT_WEB_SITE_URL ${TODO_PROJECT_WEB_SITE_URL} g" \
    -e "s TODO_PROJECT_DOCUMENTATION_URL ${TODO_PROJECT_DOCUMENTATION_URL} g" \
    -e "s TODO_PROJECT_API_DOCUMENTATION_URL ${TODO_PROJECT_API_DOCUMENTATION_URL} g" \
    AUTHORS \
    LICENSE \
    meta.make \
    README.rst


# FIX TITLES UNDERLINE LENGTH IN RESTRUCTUREDTEXT FILES #######################

PROJECT_NAME_UNDERLINE=$(echo "${TODO_PROJECT_NAME}" | tr '[:print:]' '=')

sed -i "" \
    -e "s/^====$/${PROJECT_NAME_UNDERLINE}/" \
    README.rst

