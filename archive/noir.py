#!/usr/bin/env python
"""
Interface to NOIRLab data archive.

https://astroarchive.noirlab.edu
"""
__author__ = "Alex Drlica-Wagner"
import os,sys
import logging
import requests
from datetime import date, datetime, timedelta
from dateutil.parser import parse as dateparse
import warnings
import hashlib

import numpy as np
import pandas as pd
import json

from archive.utils import date2nite, filename2expnum
from archive.database import get_desservices

URL = "https://astroarchive.noirlab.edu"
MAPPING = {'ra_min':'ra',
           'dec_min':'dec',
           'ifilter':'filter',
           }
# Sometimes the NOIRLab certificate expires... 
VERIFY=True
#VERIFY=False # only for gangstars...

def get_query(**kwargs):
    """
    Get the NOIR Lab download query dictionary.
    """
    defaults = dict(tstart=dateparse('2012-11-01'), tstop=None,
                    exptime=30,filters=('u','g','r','i','z','Y'))

    for k,v in defaults.items():
        kwargs.setdefault(k,v)

    ret = dict()
    # Output fields
    ret['outfields'] = [
        "md5sum",
        "ra_min",
        "dec_min",
        "ifilter",
        "exposure",
        "proc_type",
        "obs_type",
        "release_date",
        "caldat", # needed for nite
        #"url", 
        #"date_obs_min", # unnecessary?
        #"date_obs_max", # unnecessary?
        "proposal",
        "original_filename",  # needed for expnum
        #"archive_filename",  # unnecessary?
        "filesize",
        ]

    # Filter parameters
    ret['search'] = [
        ["instrument", 'decam'],
        ["obs_type",'object'],
        ["proc_type",'raw'],
        ["caldat", '2012-11-01','{:%Y-%m-%d}'.format(date.today())],
        ["exposure", kwargs['exptime'], 10000]
    ]

    if 'expnum' in kwargs:
        if isinstance(kwargs['expnum'],int):
            filename = 'DECam_{:08d}.fits.fz'.format(kwargs['expnum'])
        else:
            filename = 'DECam_{}.fits.fz'.format(kwargs['expnum'])
        ret['search'] += [["original_filename",filename,'endswith']]

    if 'filters' in kwargs:
        regex = '^(%s)'%('|'.join(kwargs['filters']))
        ret['search'] += [['ifilter',regex,'regex']]

    # If propid specified, get all exposures with propid
    if kwargs.get('propid',None):
        ret['search'] += [['proposal',kwargs['propid'],'exact']]
        # This should work, but it doesn't yet
        #regex = '^(%s)'%('|'.join(np.atleast_1d(kwargs['propid'])))
        #ret['search'] += [['proposal',regex,'regex']]

    if kwargs.get('tstart',None) and kwargs.get('tstop',None):
        # release date
        kwargs['tstart'] = dateparse(str(kwargs['tstart']))
        kwargs['tstop']  = dateparse(str(kwargs['tstop']))

        ret['search'] += [
            ["release_date", 
             '{:%Y-%m-%d}'.format(kwargs['tstart']), 
             '{:%Y-%m-%d}'.format(kwargs['tstop'])],
            ]

    return ret

def get_table(query=None, limit=500000, **kwargs):
    """ Get the table from NOIRLab. 

    Parameters
    ----------
    query : dictionary with query parameters
    limit : limit of exposure returned
    kwargs: passed to generate the query

    Returns
    -------
    table : pd.DataFrame
    """
    url = URL + '/api/adv_search/fasearch/?'

    if query is None:
        query = get_query(**kwargs)

    if limit is not None:
        url += 'limit={:d}'.format(limit)

    logging.info("Downloading table...")
    logging.debug(url)
    logging.debug(query)

    ret = requests.post(url,json=query, verify=VERIFY)
    data = ret.json()

    metadata = data[0]
    table = pd.read_json(json.dumps(data[1:]))

    table.rename(columns=MAPPING,inplace=True)

    expnum = create_expnum(table)
    nite = create_nite(table)
    table['expnum'] = expnum
    table['nite'] = nite

    # Could be useful...
    #url  = create_url(table)
    #table['url']  = url

    # Shouldn't be necessary, but it is on 2020-09-17...
    uid,idx,cts = np.unique(table['expnum'],return_counts=True,return_index=True)

    if np.any(cts > 1):
        logging.warning("Found %s duplicate exposures"%(cts>1).sum())

    table = table.iloc[idx]

    #table.sort_values('expnum',inplace=True)
    
    return table

def get_token(email=None,password=None):
    """
    Get authentication token

    Parameters
    ----------
    email    : email addresss
    password : password

    Returns
    -------
    token    : token string
    """
    url = URL + '/api/get_token/'

    if not email or not password:
        services = get_desservices(section='db-noirlab')
        email    = services['user']
        password = services['password']
        
    auth = dict(email=email,password=password)

    r = requests.post(url, json=auth, verify=VERIFY)
    if r.status_code == 200:
        token = r.content.decode("utf-8").strip('"')
    else:
        logging.warn(token['detail'])
        token = None
    return token

def create_expnum(data):
    """Convert original file name to exposure number."""
    col = 'original_filename'
    if not len(data): return np.array([],dtype=int)
    return filename2expnum(data[col])

def create_nite(data):
    """Convert 'caldat' to 'nite'. This is the faster option since
    it relies on NOAO to calculate the nite."""
    col = 'caldat'
    if not len(data): return np.array([],dtype='S8')
    dtype ='S%i'%(len(max(data[col], key=len)))
    nite = data[col].values.astype(dtype)
    nite = np.char.replace(nite,'-','')
    return nite

def create_url(data):
    """URL for file download"""
    col = 'md5sum'
    if not len(data): return np.array([],dtype=str)
    url = URL + '/api/retrieve/' + data[col]
    return url

def tab2npy(table):
    """ Convert table (pd.DataFrame) to numpy.recarray."""
    if isinstance(table, np.ndarray): return table
    return table.to_records(index=False)

def load_table(table):
    """ Load table from file and convert to recarray.

    Parameters
    ----------
    table : filename, DataFrame, or recarray

    Returns
    -------
    array : recarray
    """
    if isinstance(table,basestring):
        base,ext = os.path.splitext(table)
        if ext in ('.npy'):
            table = np.load(table)
        if ext in ('.csv'):
            table = pd.read_csv(table)
    return tab2npy(table)
    
def download_table(outfile, query=None, **kwargs):
    """ Download the inventory of NOAO exposures. """
    base,ext = os.path.splitext(outfile)

    table = get_table(query,**kwargs)
    logging.debug('Writing table to %s...'%outfile)
    if ext == '.npy':
        np.save(outfile,tab2npy(table))
    elif ext == '.csv':
        table.to_csv(outfile,index=False)
    else:
        msg = 'Unrecognized extension: %s'%ext
        raise Exception(msg)
    return outfile

def match_expnum(expnum,table=None):
    if table is None: 
        table = get_table(expnum=expnum)

    data = load_table(table)
    
    match = np.where(data['expnum'] == expnum)[0]
    if len(match) == 0:
        msg = "No match to exposure: %s"%expnum
        raise Exception(msg)
    elif  len(match) > 1:
        msg = "Multiple matches to exposure: %s"%expnum
        raise Exception(msg)

    return data[match[0]]

def get_file_url(expnum,table=None):
    col = 'url'
    return match_expnum(expnum,table)[col]

get_path = get_file_url

def download_file(outfile,url,headers,progress=False):
    """ Download a file from a url. """

    if not progress:
        r = requests.get(url,headers=headers,verify=VERIFY)
        with open(outfile, 'wb') as f: f.write(r.content)
    else:
        # Should be done this way, but bug in requests v2.9.1
        # https://stackoverflow.com/q/44996807/4075339
        #with requests.get(url,headers=headers,stream=True,verify=VERIFY) as r:
        r=requests.get(url,headers=headers,stream=True,verify=VERIFY)
        total_length = int(r.headers.get('content-length'))
        print("Total length: %s"%total_length)
        with open(outfile, 'wb') as f:
            dl,done = 0,0
            for chunk in r.iter_content(chunk_size=4096):
                dl += len(chunk)
                f.write(chunk)
                if int(50*dl/total_length) != done:
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )
                    sys.stdout.flush()
            print("")

    if r.status_code == 200:
        filesize = os.path.getsize(outfile)
        msg = "{} [{:.1f} MB]".format(outfile,filesize//1024**2)
        logging.info(msg)
    else:
        if os.path.exists(outfile):
            msg = "Removing failed download: {}".format(outfile)
            logging.warn(msg)
            os.remove(outfile)
        raise Exception(r.json()['message'])
    

def download_exposure(expnum,outfile=None,table=None,token=None):
    """ Download an exposure from the NOIRLab API. """

    data     = match_expnum(expnum,table)
    origsize = data['filesize']
    origmd5  = data['md5sum']

    if not outfile:
        outfile = 'DECam_{:08d}.fits.fz'.format(expnum)

    if not token: 
        token = get_token()
    
    headers = dict(Authorization=token)
    url = create_url(data)
    logging.info("Downloading %s from %s ..."%(outfile,url))
    download_file(outfile,url,headers,progress=False)

    # Check the file size 
    logging.info("Checking filesize")
    filesize = os.path.getsize(outfile)
    if origsize != filesize:
        msg = "Filesize does not match: [%i/%i]."%(filesize,origsize)
        logging.warn(msg)
        msg = "Removing failed download: {}".format(outfile)
        logging.warn(msg)
        os.remove(outfile)
        raise Exception(msg)
    logging.debug("Filesize: %s MB"%(filesize//1024**2))

    logging.info("Checking md5sum")
    md5sum = hashlib.md5(open(outfile,'rb').read()).hexdigest()
    if origmd5 != md5sum:
        msg = "File md5sum does not match: [%i/%i]."%(md5sum,origmd5)
        logging.warn(msg)
        msg = "Removing failed download: {}".format(outfile)
        logging.warn(msg)
        os.remove(outfile)
        raise Exception(msg)

    return outfile

copy_exposure = download_exposure

if __name__ == "__main__":
    from parser import Parser
    description = "Interface to NOIRLab astro data archive"
    parser = Parser(description=description)
    args = parser.parse_args()

    table = 'table.csv'
    query = get_query()
    print(query)
    print(download_table(table,query))
    
    expnum = 797795
    #print(match_expnum(expnum,table)['url'])
    download_exposure(expnum)
    
