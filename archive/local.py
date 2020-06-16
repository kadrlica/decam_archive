#!/usr/bin/env python
"""
Access to local exposure archive.
"""

import os,shutil
import subprocess
import logging
import glob
from multiprocessing import cpu_count
from collections import OrderedDict as odict

import numpy as np
import fitsio
import pandas as pd

from archive.sispi import expnum2nite
from archive.utils import filename2nite,filename2expnum
from archive import DIRNAME,BASENAME
from archive.interruptible_pool import InterruptiblePool as Pool

LOCAL_PATHS = [
    '/data/des30.b/data/DTS/src',
    '/data/des40.b/data/DTS/src',
    '/data/des41.b/data/DTS/src',
    '/data/des51.b/data/DTS/src'
    ]

# Preference should be given to fist entry
BLISS_PATHS = [
    '/data/des61.b/data/BLISS',
    '/data/des60.b/data/BLISS',
    '/data/des50.b/data/BLISS',
    ]

FILE_INFO = odict([('expnum',int),('ccdnum',int),('band','S1'),
                   ('reqnum',int),('attnum',int),('compression','S5'),
                   ('filename',object),('path',object),('filetype',object)
                   ])
FILE_INFO_DTYPE = FILE_INFO.items()                  

FILE_TYPES = odict([
        ('immask', dict(suffix='immask.fits.fz')),
        ('fullcat',dict(suffix='fullcat.fits')),
        ('psfex',  dict(suffix='sextractorPSFEX.psf')),
        ('allzp',  dict(prefix='Merg_allZP_',suffix='.csv')),
        ])

def get_nites(path=None):
    if path: paths = [path]
    else: paths = LOCAL_PATHS
    files = []
    for p in paths:
        files += glob.glob(p+'/[0-9]*[0-9]/')
        files += glob.glob(p+'/[0-9]*[0-9]/')
    files = np.atleast_1d(files)
    files.sort()
    nite   = filename2nite(files)
    return np.rec.fromarrays([nite],names=['nite'])
                             
def get_inventory(paths=None):
    paths = np.atleast_1d(paths) if paths else LOCAL_PATHS
    files = []
    # TODO: multiprocessing?
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

def get_path(expnum, nite=None, paths=None):
    if nite is None:
        nite = expnum2nite(expnum)

    PATHS = np.atleast_1d(paths) if paths else LOCAL_PATHS
    for arch in PATHS:
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

def copy_exposure(expnum,outfile=None,paths=None):
    if not outfile: outfile = '.'
    path = get_path(expnum,paths=paths)
    cmd = 'cp {} {}'.format(path,outfile)
    logging.info(cmd)
    subprocess.check_call(cmd,shell=True)
    #shutil.copy(path,outfile)
    return outfile

def link_exposure(expnum,outfile=None,paths=None):
    path = get_path(expnum,paths=paths)
    if not outfile: 
        outfile = os.path.basename(path)
    cmd = 'ln -s {} {}'.format(path,outfile)
    logging.info(cmd)
    subprocess.check_call(cmd,shell=True)
    #os.symlink(path,outfile)
    return outfile

def read_header(kwargs):
    """Wrapper around fitsio.read_header to work with Pool.map"""
    # Only compatible with python 2
    # https://stackoverflow.com/a/6062799/4075339
    try:
        return fitsio.read_header(**kwargs)
    except Exception as e:
        raise type(e)(e.message + kwargs.get('filename',''))

def read_exposure_headers(expnums, multiproc=False):
    if np.isscalar(expnums):
        expnums = [expnums]

    filepaths = [get_path(e) for e in expnums]
    kwargs = [dict(filename=f) for f in filepaths]
    if multiproc:
        p = Pool(processes=cpu_count()-2, maxtasksperchild=10)
        headers = p.map(read_header,kwargs)
        p.close()
        del p
    else:
        headers = map(read_header,kwargs)

    # Add the filename
    #filenames = np.char.rpartition(filepaths,'/')[:,-1]
    inv = parse_reduced_filepath(filepaths)
    for i,h in enumerate(headers):
        h['FILEPATH'] = filepaths[i]
        h['FILENAME'] = inv['filename'][i]

    return headers

def read_image_headers(filepaths, multiproc=False):
    #if np.isscalar(expnums):
    #    expnums = [expnums]
    # 
    #filepaths=get_image_files(expnum=expnums)

    filepaths = np.atleast_1d(filepaths).astype(str, copy=False)
    kwargs = [dict(filename=f,ext='SCI') for f in filepaths]
    if multiproc:
        p = Pool(processes=cpu_count()-2, maxtasksperchild=10)
        headers = p.map(read_header,kwargs)
        p.close()
        del p
    else:
        headers = map(read_header,kwargs)

    # Add the filename
    #filenames = np.char.rpartition(filepaths,'/')[:,-1]
    inv = parse_reduced_filepath(filepaths)
    for i,h in enumerate(headers):
        h['FILEPATH'] = filepaths[i]
        h['FILENAME'] = inv['filename'][i]

    return headers

#def get_exposure_header_info(expnums):
#    headers = read_exposure_headers(expnums)
#    df = pd.DataFrame.from_records(headers)
#    return df.to_records(index=False)

def get_reduced_exposure_paths(multiproc=False,paths=None):
    """ Get the paths to reduced exposures. """
    chunkdir = '[0-9]*00'
    expdir   = '[0-9]*[0-9]'

    PATHS = np.atleast_1d(paths) if paths else BLISS_PATHS
    path = [os.path.join(p,chunkdir,expdir) for p in PATHS]
    if multiproc:
        p = Pool(processes=2, maxtasksperchild=10)
        filepaths = p.map(glob.glob,path)
        p.close()
        del p
    else:
        filepaths = map(glob.glob,path)

    #filepaths = np.array(filepaths).flatten()
    filepaths = np.concatenate(filepaths)

    expnums=np.char.rpartition(np.char.rstrip(filepaths,'/'),'/')[:,-1].astype(int)
    inv = np.rec.fromarrays([filepaths, expnums],names=['filepath','expnum'])

    # Prefer first paths in BLISS_PATHS
    uexp,idx = np.unique(inv['expnum'],return_index=True)
    return inv[idx]

def get_reduced_files(expnum=None, prefix='', suffix='immask.fits.fz',
                      multiproc=False, paths=None):
    """ 
    Get the path to reduced files matching the pattern:
      '%{prefix}D00[0-9]*{suffix}'

    Parameters:
    -----------
    expnum : specific expnum(s) to search
    prefix : file basename prefix
    suffix : file basename suffix

    Returns:
    --------
    files  : sorted array of filepaths
    """
    inv = get_reduced_exposure_paths(multiproc=multiproc,paths=paths)

    expnum = np.atleast_1d(expnum)
    if not (len(expnum)==0 or expnum[0]==None):
        if np.any(~np.in1d(expnum,inv['expnum'])):
            msg = "Exposure(s) not found:\n"
            msg += str(expnum[~np.in1d(expnum,inv['expnum'])])
            logging.warn(msg)

        df = pd.DataFrame({'expnum':expnum})
        m = df.merge(pd.DataFrame(inv),on='expnum')
        expdir = m['filepath'].values.astype(str)
    else:
        expnum = inv['expnum']
        expdir = inv['filepath']

    regex = '%sD00[0-9]*%s'%(prefix,suffix)
    path = np.char.add(expdir,'/'+regex)

    args = zip(range(len(expdir)),expdir)
    if multiproc:
        nproc = cpu_count()-2 if isinstance(multiproc,bool) else multiproc
        p = Pool(processes=nproc, maxtasksperchild=10)
        filepaths = p.map(glob.glob,path)
        p.close()
        del p
    else:
        filepaths = map(glob.glob,path)

    filepaths = np.concatenate(filepaths)
    filepaths.sort()
    return filepaths

def get_catalog_files(**kwargs):
    kwargs.update(FILE_TYPES['fullcat'])
    return get_reduced_files(**kwargs)

def get_image_files(**kwargs):
    kwargs.update(FILE_TYPES['immask'])
    return get_reduced_files(**kwargs)

def get_psfex_files(**kwargs):
    kwargs.update(FILE_TYPES['psfex'])
    return get_reduced_files(**kwargs)

def get_zeropoint_files(**kwargs):
    kwargs.update(FILE_TYPES['allzp'])
    return get_reduced_files(**kwargs)

def fill_file_info(array, **kwargs):
    """Fill the columns of a file info array from a list of kwargs"""
    names = np.array(array.dtype.names)
    keys = kwargs.keys()
    match = np.in1d(names,keys)
    if not np.all(match):
        msg = "Unmatched columns:\n  %s"%(names[~match])
        raise ValueError(msg)

    for key,val in kwargs.items():
        array[key] = val

    # Should/can we cast the object fields to strings?


def parse_reduced_filepath(filepath):
    """Parse a reducted filepath into path,filename,compression.
    
    Parameters:
    -----------
    filepath : full path to file

    Returns:
    --------
    out : array with 'path','filename','compression'
    """
    keys = ['path','filename','compression']
    dtype = [(k,v) for k,v in FILE_INFO.items() if k in keys]
    out = np.recarray(len(filepath),dtype=dtype)

    filepath = np.atleast_1d(filepath).astype('str',copy=False)
    path,sep,basename = np.char.rpartition(filepath,'/').T
    compress = np.where(np.char.endswith(basename,'.fz'),'.fz','')
    filename = np.char.partition(basename,'.fz')[:,0]
    fill_file_info(out,path=path,filename=filename,compression=compress)
    return out

def parse_standard_file(filepath,out=None):
    """Parse the standard reduced filenames of the type:
      'D%(expnum)08d_%(band)s_%(ccdnum)02d_r%(reqnum)ip%(attnum)_*.fits.fz'
    """
    #filename = np.atleast_1d(filename)
    #path,sep,basename = np.char.rpartition(filename,'/').T
    #compress = np.where(np.char.endswith(basename,'.fz'),'.fz','')
    #fname = np.char.rstrip(basename,'.fz')
    
    fname = parse_reduced_filepath(filepath)
    stripped = np.char.lstrip(fname['filename'].astype(str,copy=False),'D')
    expnum,band,ccdnum,reqatt,ftype=np.vstack(np.char.split(stripped,'_',4)).T
    ftype = np.char.partition(ftype,'.')[:,0]
    reqnum,attnum=np.char.partition(np.char.lstrip(reqatt,'r'),'p')[:,[0,2]].T

    if out is None:
        out = np.recarray(len(filepath),dtype=FILE_INFO_DTYPE)

    fill_file_info(out, expnum=expnum, ccdnum=ccdnum, band=band, reqnum=reqnum,
                   attnum=attnum, filename=fname['filename'], 
                   path=fname['path'], filetype=ftype,
                   compression=fname['compression'])

    return out

def parse_zeropoint_file(filepath,out=None):
    """Parse the zeropoint reduced filenames of the type:
      'Merg_allZP_D%(expnum)08d_r%(reqnum)ip%(attnum).csv'
    """
    #filename = np.atleast_1d(filename)
    #path,sep,basename = np.char.rpartition(filename,'/').T
    #compress = ''
    #fname = basename

    fname = parse_reduced_filepath(filepath)
    stripped = np.char.rstrip(np.char.lstrip(fname['filename'].astype(str,copy=False),'Merg_allZP_D'),'.csv')
    expnum,reqatt=np.vstack(np.char.split(stripped,'_',1)).T
    ccdnum,band = -1,''
    ftype = 'allzp'
    reqnum,attnum=np.char.partition(np.char.lstrip(reqatt,'r'),'p')[:,[0,2]].T

    if out is None:
        out = np.recarray(len(filepath),dtype=FILE_INFO_DTYPE)

    fill_file_info(out, expnum=expnum, ccdnum=ccdnum, band=band, reqnum=reqnum,
                   attnum=attnum, filename=fname['filename'], 
                   path=fname['path'], filetype=ftype,
                   compression=fname['compression'])

    return out

def parse_psfex_file(filepath,out=None):
    """Parse the psfex reduced filenames of the type:
      'D%(expnum)08d_%(band)s_%(ccdnum)02d_r%(reqnum)ip%(attnum)_sextractorPSFEX.psf'
    """
    #filename = np.atleast_1d(filename)
    #path,sep,basename = np.char.rpartition(filename,'/').T
    #compress = ''
    #fname = basename

    fname = parse_reduced_filepath(filepath)
    stripped = np.char.lstrip(fname['filename'].astype(str,copy=False),'D')
    expnum,band,ccdnum,reqatt,ftype=np.vstack(np.char.split(stripped,'_',4)).T
    ftype = 'psfex'
    reqnum,attnum=np.char.partition(np.char.lstrip(reqatt,'r'),'p')[:,[0,2]].T

    if out is None:
        out = np.recarray(len(filepath),dtype=FILE_INFO_DTYPE)

    fill_file_info(out, expnum=expnum, ccdnum=ccdnum, band=band, reqnum=reqnum,
                   attnum=attnum, filename=fname['filename'], 
                   path=fname['path'], filetype=ftype,
                   compression=fname['compression'])

    return out


def parse_reduced_file(filename,out=None):
    """Convert reduced filename to expnum, ccdnum, reqnum, attnum.
    
    Parameters:
    -----------
    filename: String or array of strings that describe the filepath

    Returns:
    --------
    ret :  Record array of file info
    """
    scalar = np.isscalar(filename)
    filename = np.atleast_1d(filename)

    filename = filename.astype(str,copy=False)
    #if not filename.dtype.char == 'S':
    #    dtype = 'S%i'%(len(max(filename, key=len)))
    #    filename = filename.astype(dtype)

    bad = ~(np.char.endswith(filename,'.fz')|
            np.char.endswith(filename,'.fits')|
            np.char.endswith(filename,'.csv')|
            np.char.endswith(filename,'.psf')
            )
    if np.any(bad):
        msg = "Invalid file extension:"
        msg += str(filename[bad])
        raise ValueError(msg)

    if out is None: 
        out = np.recarray(len(filename),dtype=FILE_INFO_DTYPE)

    path,sep,basename = np.char.rpartition(filename,'/').T

    idx = np.arange(len(basename),dtype=int)
    zp_idx = np.where(np.char.startswith(basename,'Merg'))[0]
    psf_idx = np.where(np.char.endswith(basename,'.psf'))[0]
    other_idx = np.where(~np.in1d(idx,np.concatenate([zp_idx,psf_idx])))[0]

    if len(zp_idx):
        out[zp_idx] = parse_zeropoint_file(filename[zp_idx])
    if len(psf_idx):
        out[psf_idx] = parse_psfex_file(filename[psf_idx])
    if len(other_idx):
        out[other_idx] = parse_standard_file(filename[other_idx])

    if scalar:
        return out[0]
    return out

if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()

    print get_path(375389)
