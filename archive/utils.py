#!/usr/bin/env python
"""
Some utility functions
"""
__author__ = "Alex Drlica-Wagner"

import numpy as np
import pandas as pd
from datetime import timedelta
import warnings

def date2nite(date, exact=True):
    """ Convert a date string or datetime (array) into a nite.

    Parameters:
    -----------
    date : Input date (string or array).
    exact: If only day specified in 'date', assume it is the nite
    
    Returns:
    --------
    nite: Return nite string.
    """
    scalar = np.isscalar(date)
    date = pd.DatetimeIndex(np.atleast_1d(date))

    try:
        # Date does not have a timezone
        utc = date.tz_localize('UTC')
    except:
        # Date already has a timezone
        utc = date.tz_convert('UTC')
        
    nite = (utc - timedelta(hours=12)).strftime('%Y%m%d')

    # If only the day was specified, assume the user meant it as the nite
    if exact:
        exact = (utc == utc.normalize())
        nite[exact] = utc[exact].strftime('%Y%m%d')

    if scalar:
        return np.asscalar(nite)
    return nite

def filename2expnum(filename):
    """Convert filename to exposure number.
    
    Parameters:
    -----------
    filename: String(s) of the form 'DECam_%(expnum)08d.fits.fz'
    
    Returns:
    expnum :  Exposure number(s) extracted from filename.
    """
    scalar = np.isscalar(filename)
    filename = np.atleast_1d(filename)

    if not filename.dtype.char == 'S':
        dtype = 'S%i'%(len(max(filename, key=len)))
        filename = filename.astype(dtype)

    bad = ~(np.char.endswith(filename,'.fits.fz')|
            np.char.endswith(filename,'.fits'))
    if np.any(bad):
        msg = "Invalid file extension:"
        msg += str(filename[bad])
        raise ValueError(msg)

    basenames = np.char.rpartition(filename,'/')[:,-1]
    splitexts = np.char.strip(basenames,'.fits.fz')
    expnum = np.char.rpartition(splitexts,'_')[:,-1].astype(int)

    bad = (expnum < int(1e5)) | (expnum > int(1e8))
    if np.any(bad):
        msg = "Invalid exposure number:"
        msg += str(filename[bad])
        raise ValueError(msg)

    if scalar:
        return np.asscalar(expnum)
    return expnum

def filename2nite(filename):
    """Convert filename to exposure number.
    
    Parameters:
    -----------
    filename: Paths(s) of the form:
              '<...>/%(nite)08i/DECam_%(expnum)08d.fits.fz'
              '<...>/%(nite)08i/src/DECam_%(expnum)08d.fits.fz'
    
    Returns:
    nite :  Nite extracted from filename.
    """

    scalar = np.isscalar(filename)
    filename = np.atleast_1d(filename)

    # It would be good to do more sanity checks...
    bad = (np.char.count(filename,'/') < 2)
    if np.any(bad):
        msg = 'Invalid path:'
        msg += str(filename[bad])
        raise ValueError(msg)

    base = np.char.rpartition(filename,'/')[:,0]
    base = np.char.rstrip(base,'/src')
    nite = np.char.rpartition(base,'/')[:,-1].astype(int)

    bad = (nite < 20120000) | (nite > 20500000)
    if np.any(bad):
        msg = 'Skipping invalid nite value:'
        msg += str(filename[bad])
        warnings.warn(msg)
    
    if scalar:
        return np.asscalar(nite)
    return nite
