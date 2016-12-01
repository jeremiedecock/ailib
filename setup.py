#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO_PROJECT_NAME

# The MIT License
#
# TODO_COPYRIGHT_NOTICE
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

# Here is the procedure to submit updates to PyPI
# ===============================================
#
# 1. Register to PyPI:
#
#    $ python3 setup.py register
#
# 2. Upload the source distribution:
#
#    $ python3 setup.py sdist upload

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


# SETUP VARIABLES #############################################################

from TODO_PYTHON_PACKAGE_NAME import __version__ as VERSION

AUTHOR_NAME = 'TODO_AUTHOR_NAME'
AUTHOR_EMAIL = 'TODO_AUTHOR_EMAIL'

PYTHON_PACKAGE_NAME = 'TODO_PYTHON_PACKAGE_NAME'
PROJECT_SHORT_DESC = 'TODO_PROJECT_SHORT_DESC'
PROJECT_WEB_SITE_URL = 'TODO_PROJECT_WEB_SITE_URL'

# See :  http://pypi.python.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS = ['Development Status :: 4 - Beta',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: MIT License',
               'Natural Language :: English',
               'Operating System :: OS Independent',
               'Programming Language :: Python :: 3',
               'Topic :: Software Development :: Libraries',
               'Topic :: Software Development :: Libraries :: Python Modules',
               'Topic :: Software Development :: Libraries :: Application Frameworks']

KEYWORDS = 'TODO_PYTHON_PACKAGE_NAME'

# You can either specify manually the list of packages to include in the
# distribution or use "setuptools.find_packages()" to include them
# automatically with a recursive search (from the root directory of the
# project).
#PACKAGES = find_packages()
PACKAGES = ['TODO_PYTHON_PACKAGE_NAME']


# The following list contains all dependencies that Python will try to
# install with this project
# E.g. INSTALL_REQUIRES = ['pyserial >= 2.6']
INSTALL_REQUIRES = []


# E.g. SCRIPTS = ["examples/pyax12demo"]
SCRIPTS = []


# Entry point can be used to create plugins or to automatically generate
# system commands to call specific functions.
# Syntax: "name_of_the_command_to_make = package.module:function".
# E.g.:
#   ENTRY_POINTS = {
#     'console_scripts': [
#         'pyax12gui = pyax12.gui:run',
#     ],
#   }
ENTRY_POINTS = {}


README_FILE = 'README.rst'

def get_long_description():
    with open(README_FILE, 'r') as fd:
        desc = fd.read()
    return desc


###############################################################################

setup(author=AUTHOR_NAME,
      author_email=AUTHOR_EMAIL,
      maintainer=AUTHOR_NAME,
      maintainer_email=AUTHOR_EMAIL,

      name=PYTHON_PACKAGE_NAME,
      description=PROJECT_SHORT_DESC,
      long_description=get_long_description(),
      url=PROJECT_WEB_SITE_URL,
      download_url=PROJECT_WEB_SITE_URL, # Where the package can be downloaded

      classifiers=CLASSIFIERS,
      #license='MIT',            # Useless if license is already in CLASSIFIERS
      keywords=KEYWORDS,

      packages=PACKAGES,
      include_package_data=True, # Use the MANIFEST.in file

      install_requires=INSTALL_REQUIRES,
      #platforms=['Linux'],
      #requires=['pyserial'],

      scripts=SCRIPTS,
      entry_points=ENTRY_POINTS,

      version=VERSION)
