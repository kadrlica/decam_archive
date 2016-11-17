#!/usr/bin/env python
"""
Download exposures from DESDM desar2.

Expects DESDM username/password to be stored in .netrc file.
"""

__author__ = "Alex Drlica-Wagner"
import os
import subprocess
import logging

from archive import DIRNAME, BASENAME
from archive.sispi import expnum2nite

DESAR2_URL = "https://desar2.cosmology.illinois.edu/DESFiles/desarchive/DTS/raw"
DESAR2_WGET = "wget -t 50 --retry-connrefused --no-check-certificate --waitretry 30 --progress=dot -e dotbytes=4M --timeout 120 -O {outfile} {url} || rm -f {outfile}"

WGET_EXP = DESAR2_WGET
WGET_NITE = "wget  -X . -r -A DECam_\*fits.fz  -np --level=2 --no-check-certificate -N -nH --cut-dirs=4 --progress=dot -e dotbytes=4M {url}/{nite}"


def get_path(expnum):
    nite = expnum2nite(expnum)
    path = os.path.join(DESAR2_URL,DIRNAME,BASENAME)
    return path.format(nite=nite,expnum=expnum)

def download_exposure(expnum,outfile=None):
    path = get_path(expnum)
    if not outfile:
        outfile = os.path.basename(path)
    cmd = DESAR2_WGET.format(url=path,outfile=outfile)
    logging.info(cmd)
    subprocess.check_call(cmd,shell=True)
    return outfile

copy_exposure = download_exposure

def download_nite(nite,outdir=None):
    if not outdir: outdir = './'
    os.chdir(outdir)
    cmd = WGET_NITE
    cmd.format(url=DESAR2_URL,nite=nite)
    loggging.info(cmd)
    subprocess.check_call(cmd,shell=True)
    return outdir


if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()

    print get_path(335589)
