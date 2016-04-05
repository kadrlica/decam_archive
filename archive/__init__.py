#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

DIRNAME = '{nite:>08}'
BASENAME = 'DECam_{expnum:>08}.fits.fz'
