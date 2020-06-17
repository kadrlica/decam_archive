#!/usr/bin/env python
"""
Interface to NOIRLab data archive.

https://astroarchive.noao.edu
"""
__author__ = "Alex Drlica-Wagner"
import os,sys
import logging
import requests
from datetime import date, datetime, timedelta
from dateutil.parser import parse as dateparse
import warnings

import numpy as np
import pandas as pd
import json

from archive.utils import date2nite, filename2expnum
from archive.database import get_desservices

URL = "https://astroarchive.noao.edu"
MAPPING = {'ra_min':'ra',
           'dec_min':'dec',
           'ifilter':'filter',
           }
    
def get_noir_query(**kwargs):
    """
    Get the NOIR Lab download query dictionary.
    """
    #filters=('u','g','r','i','z','Y'),
    defaults = dict(tstart=dateparse('2012-11-01'), tstop=date.today(),
                    exptime=30, filter='')

    for k,v in defaults.items():
        kwargs.setdefault(k,v)

    kwargs['tstart'] = dateparse(str(kwargs['tstart']))
    kwargs['tstop']  = dateparse(str(kwargs['tstop']))

    ret = dict()
    # Output fields
    ret['outfields'] = [
        "md5sum",
        "ra_min",
        "dec_min",
        "ifilter",
        "proc_type",
        "obs_type",
        "release_date",
        "caldat",
        "date_obs_min", # unnecessary?
        "date_obs_max", # unnecessary?
        "proposal",
        "original_filename", 
        "archive_filename",  # unnecessary?
        "filesize",
        ]

    # Filter parameters
    ret['search'] = [
        ["instrument", 'decam'],
        ["obs_type",'object'],
        ["proc_type",'raw'],
        ["release_date", '{:%Y-%m-%d}'.format(kwargs['tstart']), '{:%Y-%m-%d}'.format(kwargs['tstop'])],
        ["caldat", '2012-11-01', '{:%Y-%m-%d}'.format(date.today())],
        ['exposure', kwargs['exptime'], 10000],
    ]

    if 'expnum' in kwargs:
        if isinstance(kwargs['expnum'],int):
            filename = 'DECam_{:08d}.fits.fz'.format(kwargs['expnum'])
        else:
            filename = 'DECam_{}.fits.fz'.format(kwargs['expnum'])
        ret['search'] += [["original_filename",filename,'endswith']]

    if 'filter' in kwargs:
        ret['search'] += [['ifilter',kwargs['filter'],'startswith']]

    return ret

def get_table(query=None, limit=250000, **kwargs):
    """ Get the VOTable from NOAO. """
    url = URL + '/api/adv_search/fasearch/?'

    if query is None:
        query = get_noir_query(**kwargs)

    if limit is not None:
        url += 'limit={:d}'.format(limit)

    logging.info("Downloading table...")
    ret = requests.post(url,json=query)
    table = pd.read_json(json.dumps(ret.json()))

    table.rename(MAPPING,inplace=True)

    expnum = create_expnum(table)
    nite = create_nite(table)
    table['expnum'] = expnum
    table['nite'] = nite

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
        services = get_desservices(section='db-noir')
        email    = services['user']
        password = services['password']
        
    auth = dict(email=email,password=password)

    # May need verify=False if certificate invalid
    r = requests.post(url, json=auth, verify=True)
    if r.status_code == 200:
        token = r.content.decode("utf-8")
    else:
        logging.warn(token['detail'])
        token = None
    return token

def create_expnum(data):
    """Convert original file name to exposure number"""
    col = 'original_filename'
    if not len(data): return np.array([],dtype=int)
    return filename2expnum(data[col])

def create_nite_safe(data):
    """Convert 'date_obs' to nite. This is safer (using one
    'date2nite' converter) but slower option."""
    col = 'caldat'
    if not len(data): return np.array([],dtype=int)
    return date2nite(data[col])

def create_nite_fast(data):
    """Convert 'caldat' to 'nite'. This is the faster option since
    it relies on NOAO to calculate the nite."""
    col = 'caldat'
    if not len(data): return np.array([],dtype='S8')
    dtype ='S%i'%(len(max(data[col], key=len)))
    nite = data[col].values.astype(dtype)
    nite = np.char.replace(nite,'-','')
    return nite

create_nite = create_nite_fast

def tab2npy(table):
    if isinstance(table, np.ndarray): return table
    return table.to_records(index=False)

def load_table(table):
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
    if ext == '.npy':
        np.save(outfile,tab2npy(table))
    elif ext == '.csv':
        table.to_csv(outfile,index=False)
    else:
        msg = 'Unrecognized extension: %s'%ext
        raise Exception(msg)
    return outfile

def match_expnum(expnum,table=None):
    if not table: 
        table = get_table(expnum=expnum)

    data = load_table(table)
    
    match = np.where(data['expnum'] == expnum)[0]
    if len(match) == 0 or len(match) > 1:
        msg = "No unique match to exposure: %s"%expnum
        raise Exception(msg)

    return data[match[0]]

def get_file_url(expnum,table=None):
    col = 'url'
    return match_expnum(expnum,table)[col]

get_path = get_file_url

def download_exposure(expnum,outfile=None,table=None,token=None):
    data = match_expnum(expnum,table)
    path = data['url']
    origsize = data['filesize']

    if not outfile:
        outfile = 'DECam_{:08d}.fits.fz'.format(expnum)

    if not token: 
        token = get_token()

    
    headers = dict(Authorization=token)
    with open(outfile, 'wb') as f:
        logging.info("Downloading %s..."%outfile)
        r = requests.get(data['url'],headers=headers,stream=True)
        total_length = r.headers.get('content-length')

        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in r.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()         
            sys.stdout.write('\n')

    if r.status_code == 200:
        filesize = os.path.getsize(outfile)
        msg = "{} [{:.1f} MB]".format(outfile,filesize//1024**2)
        logging.info(msg)
    else:
        raise Exception(r.json()['message'])

    # Check the file size (might be unnecessary with wget)
    filesize = os.path.getsize(outfile)
    if origsize != filesize:
        msg = "Filesize does not match: [%i/%i]."%(filesize,origsize)
        os.remove(outfile)
        raise Exception(msg)
    logging.debug("Filesize: %s MB"%(filesize//1024**2))
    return outfile

copy_exposure = download_exposure


if __name__ == "__main__":
    from parser import Parser
    description = "Interface to NOIR lab data archive"
    parser = Parser(description=description)
    args = parser.parse_args()

    #table = 'table.csv'
    #query = get_noir_query()
    #print(query)
    #print(download_table(table,get_noir_query()))
    
    expnum = 797795
    #print(match_expnum(expnum,table)['url'])
    download_exposure(expnum)
    
