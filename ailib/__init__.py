"""AILib

AILib is a set of open source frameworks for Artificial Intelligence
(mostly machine learning and optimization).

This contains (among others):

-  a blackbox non linear noisy optimization framework;
-  a machine learning framework;
-  a multistage optimization and Markov Decision Process framework
   (Markov Decision Processes).

Note:

    This project is in beta stage.

Viewing documentation using IPython
-----------------------------------
To see which functions are available in `ailib`, type ``ailib.<TAB>`` (where
``<TAB>`` refers to the TAB key), or use ``ailib.*optimize*?<ENTER>`` (where
``<ENTER>`` refers to the ENTER key) to narrow down the list.  To view the
docstring for a function, use ``ailib.optimize?<ENTER>`` (to view the
docstring) and ``ailib.optimize??<ENTER>`` (to view the source code).
"""

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
# X.Y
# X.Y.Z # For bugfix releases
#
# Admissible pre-release markers:
# X.YaN # Alpha release
# X.YbN # Beta release
# X.YrcN # Release Candidate
# X.Y # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '0.3.dev1'

def get_version():
    return __version__

# The following lines are temporary commented to avoid BUG#2 (c.f. BUGS.md)
#from . import ml
#from . import mdp
#from . import optimize
#from . import signal
