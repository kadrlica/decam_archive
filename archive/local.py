#!/usr/bin/env python
"""
Access to local exposure archive.
"""

import os,shutil
import subprocess
import logging
import glob

import numpy as np
import fitsio
import pandas as pd

from archive.sispi import expnum2nite
from archive.utils import filename2nite,filename2expnum
from archive import DIRNAME,BASENAME

LOCAL_PATHS = [
    '/data/des30.b/data/DTS/src',
    '/data/des40.b/data/DTS/src',
    '/data/des51.b/data/DTS/src'
    ]

BLISS_PATHS = [
    '/data/des50.b/data/BLISS',
    ]

def get_inventory(path=None):
    if path: paths = [path] 
    else: paths = LOCAL_PATHS
    files = []
    for p in paths:
        files += glob.glob(p+'/[0-9]*[0-9]/DECam_[0-9]*[0-9].fits.fz')
        files += glob.glob(p+'/[0-9]*[0-9]/src/DECam_[0-9]*[0-9].fits.fz')
    files = np.atleast_1d(files)
    files.sort()
    nite   = filename2nite(files)
    expnum = filename2expnum(files)
    #data = np.array(files)
    #data = np.sort(data)
    #base,sep,exp = np.char.rpartition(data,'/').T
    #nite = np.char.rpartition(base,'/')[:,-1].astype(int)
    #expnum = np.char.partition(np.char.rpartition(exp,'_')[:,-1],'.')[:,0].astype(int)
    return np.rec.fromarrays([files, nite, expnum],
                             names=['filename','nite','expnum'])

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
            return os.path.abspath(filename)
    msg = "Exposure not found locally: %s\n%s"%(expnum,filename)
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

def read_header(expnum):
    return fitsio.read_header(get_path(expnum))

def read_headers(expnums, multiproc=False):
    if np.isscalar(expnums):
        expnums = [expnums]

    if multiproc:
        from multiprocessing import Pool,  cpu_count
        p = Pool(processes=cpu_count()-2, maxtasksperchild=10)
        headers = p.map(read_header,expnums)
        p.close()
        del p
    else:
        headers = map(read_header,expnums)

    return headers

def get_header_info(expnums):
    headers = read_headers(expnums)
    df = pd.DataFrame.from_records(headers)
    return df.to_records(index=False)

def get_reduced_files(path=None,prefix='',suffix='immask.fits.fz'):
    """ Get the path to reduced files. """
    if path: paths = [path] 
    else: paths = BLISS_PATHS
    files = []
    for p in paths:
        files += glob.glob(p+'/[0-9]*00/[0-9]*[0-9]/%sD00[0-9]*%s'%(prefix,suffix))
    files = np.atleast_1d(files)
    files.sort()
    return files
    
def get_catalog_files(path=None):
    return get_reduced_files(path,suffix='fullcat.fits')

def get_immask_files(path=None):
    return get_reduced_files(path,suffix='immask.fits.fz')

def get_psf_files(path=None):
    return get_reduced_files(path,suffix='sextractorPSFEX.psf')

def get_zeropoint_files(path=None):
    return get_reduced_files(path,prefix='Merg_allZP_',suffix='.csv')

def parse_reduced_file(filename):
    """Convert reduced filename to expnum, ccdnum, reqnum, attnum.
    
    Parameters:
    -----------
    filename: String(s) of the form:
      'D%(expnum)08d_%(band)s_%(ccdnum)02d_r%(reqnum)ip%(attnum)_*.fits.fz'
    
    Returns:
    ret :  Record array of file info
    """
    scalar = np.isscalar(filename)
    filename = np.atleast_1d(filename)

    if not filename.dtype.char == 'S':
        dtype = 'S%i'%(len(max(filename, key=len)))
        filename = filename.astype(dtype)

    bad = ~(np.char.endswith(filename,'.fits.fz')|
            np.char.endswith(filename,'.fits')|
            np.char.endswith(filename,'.csv'))
    if np.any(bad):
        msg = "Invalid file extension:"
        msg += str(filename[bad])
        raise ValueError(msg)

    dtype = [('expnum',int),('ccdnum',int),('band','S1'),
             ('reqnum',int),('attnum',int),('compression','S5'),
             ('filename',object),('path',object),('filetype',object)]
             
    out = np.recarray(len(filename),dtype=dtype)
    
    path,sep,basename = np.char.rpartition(filename,'/').T
    compress = np.where(np.char.endswith(basename,'.fz'),'.fz','')
    fname = np.char.rstrip(basename,'.fz')
    stripped = np.char.lstrip(fname,'D')
    expnum,band,ccdnum,reqatt,ftype=np.vstack(np.char.split(stripped,'_',4)).T
    reqnum,attnum=np.char.partition(np.char.lstrip(reqatt,'r'),'p')[:,[0,2]].T
    ftype = np.char.partition(ftype,'.')[:,0]

    out['expnum'] = expnum
    out['ccdnum'] = ccdnum
    out['band']   = band
    out['reqnum'] = reqnum
    out['attnum'] = attnum
    out['filename'] = fname
    out['path'] = path
    out['filetype'] = ftype
    out['compression'] = compress

    if scalar:
        return out[0]
    return out


if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()

    print get_path(375389)
