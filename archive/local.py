#!/usr/bin/env python
"""
Access to local exposure archive.
"""

import os,shutil
import subprocess
import logging
import glob

import numpy as np

from database import expnum2nite
from archive import DIRNAME,BASENAME

LOCAL_PATHS = [
    '/data/des30.b/data/DTS/src',
    '/data/des40.b/data/DTS/src',
    '/data/des51.b/data/DTS/src'
    ]

def get_inventory(path=None):
    if path: paths = [path] 
    else: paths = LOCAL_PATHS
    files = []
    for p in paths:
        files += glob.glob(p+'/*/DECam_*.fits.fz')
        files += glob.glob(p+'/*/src/DECam_*.fits.fz')
    data = np.array(files)
    data = np.sort(data)
    base,sep,exp = np.char.rpartition(data,'/').T
    nite = np.char.rpartition(base,'/')[:,-1].astype(int)
    expnum = np.char.partition(np.char.rpartition(exp,'_')[:,-1],'.')[:,0].astype(int)
    return data, nite, expnum


def get_path(expnum):
    nite = expnum2nite(expnum)
    for arch in LOCAL_PATHS:
        filename = os.path.join(arch,DIRNAME,BASENAME)
        filename = filename.format(nite=nite,expnum=expnum)
        logging.debug("Looking for: %s"%filename)
        if os.path.exists(filename):
            return filename
        filename = os.path.join(arch,DIRNAME,'src',BASENAME)
        filename = filename.format(nite=nite,expnum=expnum)
        logging.debug("Looking for: %s"%filename)
        if os.path.exists(filename):
            return filename
    msg = "Exposure not found locally"
    raise Exception(msg)

def copy_exposure(expnum,outfile=None):
    if not outfile: outfile = '.'
    path = get_path(expnum)
    cmd = 'cp {} {}'.format(path,outfile)
    logging.info(cmd)
    subprocess.check_call(cmd,shell=True)
    #shutil.copy(path,outfile)
    return outfile

def link_exposure(expnum,outfile=None):
    path = get_path(expnum)
    if not outfile: 
        outfile = os.path.basename(path)
    cmd = 'ln -s {} {}'.format(path,outfile)
    logging.info(cmd)
    subprocess.check_call(cmd,shell=True)
    #os.symlink(path,outfile)
    return outfile

if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()

    print get_path(375389)
