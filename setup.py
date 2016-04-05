#!/usr/bin/env python
import sys
import os
try: from setuptools import setup
except ImportError: from distutils.core import setup
import versioneer

NAME = 'decam_archive'
HERE = os.path.abspath(os.path.dirname(__file__))
CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: Python
Natural Language :: English
Topic :: Scientific/Engineering
"""

def read(filename):
    return open(os.path.join(HERE,filename)).read()

setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url='https://github.com/kadrlica/decam_archive',
    author='Alex Drlica-Wagner',
    author_email='kadrlica@fnal.gov',
    scripts = ['bin/'],
    install_requires=[
        'python >= 2.7.0',
        'psycopg2 >= 2.4.6',
    ],
    packages=['archive'],
    package_data={}
    description="File archive for DECam exposures.",
    long_description=read('README.md'),
    platforms='any',
    keywords='astronomy',
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f]
)
