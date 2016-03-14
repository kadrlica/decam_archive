#!/usr/bin/env python
import sys
import os
try: from setuptools import setup
except ImportError: from distutils.core import setup

from standalone.get_version import get_version, write_version_py

NAME = 'desdm_standalone'
HERE = os.path.abspath(os.path.dirname(__file__))
CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: Python
Natural Language :: English
Topic :: Scientific/Engineering
"""
VERSION = get_version()
write_version_py(version=VERSION)

def read(filename):
    return open(os.path.join(HERE,filename)).read()

setup(
    name=NAME
    version=VERSION,
    url='https://bitbucket.org/kadrlica/desdm_standalone',
    author='Alex Drlica-Wagner',
    author_email='kadrlica@fnal.gov',
    scripts = ['bin/'],
    install_requires=[
        'python >= 2.7.0',
    ],
    packages=['standalone'],
    package_data={'standalone':['config/*.cfg']}
    description="Standalone DESDM processing pipeline.",
    long_description=read('README.md'),
    platforms='any',
    keywords='astronomy',
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f]
)
