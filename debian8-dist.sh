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

# LOAD VARIABLES ##############################################################

source ./meta.sh

###############################################################################

VERSION=$(python -c "print(__import__('${PYTHON_PACKAGE_NAME}').__version__)")
DIST_DIR=dist

###############################################################################

rm -rfv debian

# TODO
mkdir -p                     debian/usr/local/lib/python3.0/dist-packages
cp -r ${PYTHON_PACKAGE_NAME} debian/usr/local/lib/python3.0/dist-packages
chmod 644                    $(find debian/usr/local/lib -type f)

mkdir -p      "debian/usr/share/doc/${PYTHON_PACKAGE_NAME}/"
cp LICENSE    "debian/usr/share/doc/${PYTHON_PACKAGE_NAME}/copyright"
chmod 644     "debian/usr/share/doc/${PYTHON_PACKAGE_NAME}/copyright"

mkdir -p debian/DEBIAN

# section list : http://packages.debian.org/stable/
cat > debian/DEBIAN/control << EOF
Package: ${PYTHON_PACKAGE_NAME}
Version: ${VERSION}
Section: libs
Priority: optional
Maintainer: ${AUTHOR_NAME} <${AUTHOR_EMAIL}>
Architecture: all
Depends: python (>= 3.0)
Description: ${PROJECT_SHORT_DESC}
EOF

fakeroot dpkg-deb -b debian

mkdir -p "${DIST_DIR}"
mv debian.deb "${DIST_DIR}/${PYTHON_PACKAGE_NAME}_${VERSION}_all.deb"
