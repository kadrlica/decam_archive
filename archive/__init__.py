#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
__version__ = "1.0"

DIRNAME = '{nite:>08}'
BASENAME = 'DECam_{expnum:>08}.fits.fz'



if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()
