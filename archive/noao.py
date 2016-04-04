#!/usr/bin/env python
"""
Interface to NOAO data archive.

http://archive.noao.edu/search/query
"""

import os
import re
from datetime import date, datetime
from dateutil.parser import parse as dateparse
import time
import subprocess
import logging
import requests
from io import StringIO, BytesIO

import numpy as np
import numpy.lib.recfunctions as recfuncs
import astropy.io.votable as vot

NOAO_URL = "http://archive.noao.edu/search/"
NOAO_QUERY = """
SELECT {columns}
FROM voi.siap 
WHERE (telescope = 'ct4m' AND instrument = 'decam')
AND (proctype = 'RAW' AND obstype = 'object')
AND (start_date >= '2012-11-01')
AND (release_date between '{tstart:%Y-%m-%d}' and '{tstop:%Y-%m-%d}')
AND left(filter, 1) in ({filters})
AND exposure >= {exptime}
AND dtacqnam like '%DECam_{expnum}.fits.fz'
ORDER BY date_obs ASC LIMIT {limit}
"""
NOAO_CURL = "curl {certificate} -k --show-error --retry 7 --output {outfile} {url}"
NOAO_WGET = "wget {certificate} -t 50 --retry-connrefused --waitretry 30 --progress=dot -e dotbytes=4M --timeout 30 -O {outfile} {url} || rm -f {outfile}"

# An NOAO SSL certificate can be generated here:
# https://portal-nvo-tmp.noao.edu/home/contrib
# Add to wget with `--certificate {cert}`
# Expires every two weeks...
NOAO_CERT = '/home/s1/kadrlica/projects/decam_archive/data/certificates/drlicawagnera_20160314.cert'

def get_noao_query(**kwargs):
    kwargs = get_noao_query_kwargs(**kwargs)
    return NOAO_QUERY.format(**kwargs)

def get_noao_query_kwargs(**kwargs):
    """
    Get the NOAO download query.
    """
    # Some columns are required
    required = [
        'reference', 
        'release_date', 
        'start_date', 
        'filesize', 
        'dtpropid', 
        'md5sum'
    ]

    defaults = dict(tstart=dateparse('2012-11-01'), tstop=date.today(),
                    exptime=60,filters=('g','r','i','z','Y'),
                    limit=250000,expnum='%')

    defaults['columns'] = [
        'reference', 
        'dtpropid', 
        'release_date', 
        'start_date', 
        'date_obs', 
        'instrument', 
        'ra', 
        'dec', 
        'filter', 
        'exposure', 
        'obstype', 
        'proctype', 
        'dtacqnam AS original_file', 
        'reference AS archive_file',
        'filesize',
        ]

    for k,v in defaults.items():
        kwargs.setdefault(k,v)

    kwargs['columns'] = map(str.lower,kwargs['columns'])
    kwargs['columns'] += [c for c in required if c not in kwargs['columns']]

    kwargs['tstart'] = dateparse(str(kwargs['tstart']))
    kwargs['tstop']  = dateparse(str(kwargs['tstop']))

    if not isinstance(kwargs['columns'],basestring):
        kwargs['columns'] = ','.join(kwargs['columns'])
    if not isinstance(kwargs['filters'],basestring):
        kwargs['filters'] = ','.join(["'%s'"%f for f in kwargs['filters']])
    if isinstance(kwargs['expnum'],int):
        kwargs['expnum'] = '{expnum:08d}'.format(**kwargs)

    return kwargs

def get_csrf_token(session, url=None):
    if not url:
        #http://archive.noao.edu/search/query
        url = os.path.join(NOAO_URL,'query')
    response = session.get(url)
    pattern = 'meta content="(.*)" name="csrf-token"'
    token = re.search(pattern,str(response.content)).group(1)
    return token

def get_certificate(username=None,password=None):
    if username is None or password is None:
        cert = None
    # Actually get the certificate...
    return cert

def request_votable(query=None):
    """
    Get the inventory of NOAO exposures
    """
    if logging.getLogger().getEffectiveLevel <= logging.DEBUG:
        import httplib
        httplib.HTTPConnection.debuglevel = 1

    session = requests.session()
    headers = {'X-CSRF-Token':get_csrf_token(session)} #token

    ### Set the session by hand (for debugging)
    #headers = {'X-CSRF-Token' : "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"}
    #cookies = {"_session_id":"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"}
    #session.cookies = requests.utils.cookiejar_from_dict(cookies)

    logging.debug('\n')
    for k,v in headers.items():
        logging.debug('%s: %s'%(k,v))
    logging.debug('Cookies:')
    for k,v in session.cookies.items():
        logging.debug('  %s: %s'%(k,v))
    logging.debug('\n')

    # Setup the query content to get VOTable
    #http://archive.noao.edu/search/query_content
    url = os.path.join(NOAO_URL,'query_content')
    response = session.get(url,headers=headers)
    response.raise_for_status()

    logging.debug('\n'+query)
    if not query: query = get_noao_query()
        
    #http://archive.noao.edu/search/send_advanced_query
    url = os.path.join(NOAO_URL,'send_advanced_query')
    body = dict(reset_datagrid='true',advanced_sql=query)
    response = session.post(url,data=body,headers=headers)
    response.raise_for_status()
 
    logging.debug('\n %s %s \n'%(response.status_code, response.text))

    #http://archive.noao.edu/search/records_as_votable
    url = os.path.join(NOAO_URL,"records_as_votable")
    params = dict(sort_key='date_obs',sort_order='ASC',
                  coords_format='decimal_degrees',datagrid_name='All',
                  all_rows='true')
    response = session.get(url,params=params,headers=headers)
    response.raise_for_status()

    logging.info('Downloaded VOTable with %s rows'%len(response.content.split('/n')))
    return response.content

def get_votable(query):
    """ Get the VOTable from NOAO. """
    data = request_votable(query)
    
    fileobj = BytesIO()
    fileobj.write(data)
    votable = vot.parse_single_table(fileobj)

    return votable

def download_votable(outfile, query=None, **kwargs):
    """ Download the inventory of NOAO exposures. """
    base,ext = os.path.splitext(outfile)

    if ext == '.npy':
        votable = get_votable(query)
        np.save(outfile,vot2npy(votable))
    else:
        votable = request_votable(query)
        with open(outfile,'w') as out:
            out.write(votable)
    return outfile

def load_votable(votable):
    if isinstance(votable,basestring):
        base,ext = os.path.splitext(votable)
        if ext in ('.npy'):
            votable = np.load(votable)
        if ext in ('.vot'):
            votable = vot.parse_single_table(votable)
    return vot2npy(votable)

def create_expnum(data):
    """Convert original file name to exposure number"""
    if not len(data): return np.array([],dtype='S8')
    col = 'original_file'
    dtype ='S%i'%(len(max(data[col], key=len)))
    filenames = data[col].data.astype(dtype)
    basenames = np.char.rpartition(filenames,'/')[:,-1]
    splitexts = np.char.strip(basenames,'.fits.fz')
    expnum = np.char.rpartition(splitexts,'_')[:,-1].astype(int)
    return expnum

def create_nite(data):
    """Convert start_date to nite"""
    if not len(data): return np.array([],dtype='S8')
    col = 'start_date'
    dtype ='S%i'%(len(max(data[col], key=len)))
    nite = data[col].data.astype(dtype)
    nite = np.char.replace(nite,'-','')
    return nite

def vot2npy(votable):
    if isinstance(votable, np.ndarray):
        return votable

    data = votable.array
    expnum = create_expnum(data)
    nite = create_nite(data)
    out = np.empty(len(data),dtype=data.dtype.descr+[('expnum',int),('nite',int)])
    out[:] = data[:]
    out['expnum'] = expnum
    out['nite'] = nite
    return out

def match_expnum(expnum,votable=None):
    if not votable: 
        query=get_noao_query(expnum=expnum)
        votable = get_votable(query=query)

    data = load_votable(votable)
    
    match = np.where(data['expnum'] == expnum)[0]
    if len(match) == 0 or len(match) > 1:
        msg = "No unique match to exposure: %s"%expnum
        raise Exception(msg)

    return data[match[0]]

def get_file_url(expnum,votable=None):
    return match_expnum(expnum,votable)['reference']

get_path = get_file_url

def retry(cmd, retry=25):
    for i in range(retry):
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logging.info("\n--%s-- Attempt %i..."%(now,i+1))
            return subprocess.check_call(cmd,shell=True)
        except Exception, e:
            logging.warning(e)
            sleep = i*30
            logging.info("Sleeping %is..."%sleep)
            time.sleep(sleep)
    else:
        raise Exception("Failed to execute command.")

def download_exposure(expnum,outfile=None,votable=None,certificate=None):
    data = match_expnum(expnum,votable)
    path = data['reference']
    origsize = data['filesize']

    if not outfile:
        outfile = 'DECam_{expnum:08d}.fits.fz'.format(expnum=expnum)

    tool = 'curl'
    if tool == 'wget':
        cert = '--certificate %s'%certificate if certificate else ''
        cmd = NOAO_WGET
    elif tool == 'curl':
        cert = '--cert %s'%certificate if certificate else ''
        cmd = NOAO_CURL
    cmd = cmd.format(url=path,outfile=outfile,certificate=cert)

    logging.info(cmd)
    retry(cmd,retry=25)
             
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
    description = "Interface to NOAO data archive"
    parser = Parser(description=description)
    args = parser.parse_args()

    votable = 'test.vot'
    query = get_noao_query()
    print query
    print download_votable(votable,get_noao_query())
    
    print match_expnum(335589,votable)['reference']
    download_exposure(335589)
    
