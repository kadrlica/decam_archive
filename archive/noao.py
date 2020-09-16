#!/usr/bin/env python
"""
Interface to NOAO data archive.

http://archive.noao.edu/search/query
"""

import os
import re
from datetime import date, datetime, timedelta
from dateutil.parser import parse as dateparse
import time
import subprocess
import logging
import requests
from io import StringIO, BytesIO

import numpy as np
import numpy.lib.recfunctions as recfuncs
import astropy.io.votable as vot

from archive.utils import date2nite, filename2expnum, retry
from archive.utils import get_datadir,get_datafile

#NOAO_URL = "http://archive.noao.edu/search/"
#--AND (start_date >= '2012-11-01') -- Start of SV

NOAO_URL = "http://archive1.dm.noao.edu/search/"
NOAO_QUERY = """
SELECT {columns}
FROM voi.siap 
WHERE (telescope = 'ct4m' AND instrument = 'decam')
AND (proctype = 'RAW' AND obstype = 'object')
AND (start_date >= '{tstart:%Y-%m-%d}')
AND ( (release_date between '{tstart:%Y-%m-%d}' and '{tstop:%Y-%m-%d}') {propids})
AND left(filter, 1) in ({filters})
AND exposure >= {exptime}
AND dtacqnam like '%DECam_{expnum}.fits.fz'
ORDER BY date_obs ASC LIMIT {limit}
"""

NOAO_EXPNUM_QUERY = """
SELECT {columns}
FROM voi.siap 
WHERE (telescope = 'ct4m' AND instrument = 'decam')
AND dtacqnam like '%DECam_{expnum}.fits.fz'
ORDER BY date_obs ASC LIMIT {limit}
"""

NOAO_CURL = "curl {certificate} -k --show-error --retry 0 --output {outfile} {url}"
NOAO_WGET = "wget {certificate} -t 50 --retry-connrefused --waitretry 30 --progress=dot -e dotbytes=4M --timeout 30 -O {outfile} {url} || rm -f {outfile}"

# An NOAO SSL certificate can be generated here:
# https://portal-nvo-tmp.noao.edu/home/contrib
# https://archive.noao.edu/security/get_user_certificate
# Add to wget with `--certificate {cert}`
# Expires every two weeks...
# Make sure certificate not world readable:
# > chmod go-r {cert}
CERTDIR='/data/des51.b/data/DTS/src/decam_archive/certificates'
NOAO_CERT = os.path.join(CERTDIR,'drlicawagnera-20200130.cert')

# Corrupted or otherwise bad exposures that shouldn't be downloaded
BLACKLIST = np.loadtxt(get_datafile('blacklist.txt'))

# My propids
PROPIDS = np.loadtxt(get_datafile('propid.txt'),dtype=str)

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
        'md5sum',
        'noao_id' # only required when logged in
    ]
    # Start of DECam is '2012-11-01', but due to query limits, start later
    defaults = dict(tstart=dateparse('2017-01-01'), 
                    tstop=date.today() - timedelta(1),
                    exptime=30,filters=('u','g','r','i','z','Y'),
                    limit=250000,expnum='%',
                    propids=PROPIDS
                    )
                  

    defaults['columns'] = [
        'reference', 
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

    if not kwargs['propids']: 
        kwargs['propids'] = ''
    else:
        if not isinstance(kwargs['propids'],basestring):
            kwargs['propids'] = ','.join(["'%s'"%p for p in kwargs['propids']])
        kwargs['propids'] = 'OR dtpropid in (%s)'%(kwargs['propids'])

    return kwargs

def get_csrf_token(session, url=None):
    #http://archive1.dm.noao.edu/search/query
    if not url: url = os.path.join(NOAO_URL,'query')
    logging.debug(url)
    response = session.get(url)
    pattern = 'meta content="(.*)" name="csrf-token"'
    token = re.search(pattern,str(response.content)).group(1)
    return token

def get_certificate(username=None,password=None):
    if username is None or password is None:
        cert = None
    # Actually get the certificate...
    raise Exception('Automated certificate retrieval not implemented')
    return cert

def request_votable(query=None):
    """
    Get the inventory of NOAO exposures
    """
    if logging.getLogger().getEffectiveLevel <= logging.DEBUG:
        import httplib
        #httplib.HTTPConnection.debuglevel = 2
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
    #http://archive1.dm.noao.edu/search/query_content
    url = os.path.join(NOAO_URL,'query_content')
    logging.debug(url)
    response = session.get(url,headers=headers)
    response.raise_for_status()
    #logging.debug(response.text)

    if not query: query = get_noao_query()
    logging.debug('\n'+query)

    #http://archive1.dm.noao.edu/search/send_advanced_query
    url = os.path.join(NOAO_URL,'send_advanced_query')
    body = dict(reset_datagrid='true',advanced_sql=query)
    logging.debug(url)
    #logging.debug(session.get(url,headers=headers).text)
    response = session.post(url,data=body,headers=headers)
    logging.debug(response.text)
    response.raise_for_status()
 
    logging.debug('\n %s %s \n'%(response.status_code, response.text))

    #http://archive1.dm.noao.edu/search/records_as_votable
    url = os.path.join(NOAO_URL,"records_as_votable")
    params = dict(sort_key='date_obs',sort_order='ASC',
                  coords_format='decimal_degrees',datagrid_name='All',
                  all_rows='true')
    logging.debug(url)
    response = session.get(url,params=params,headers=headers)
    response.raise_for_status()

    logging.info('GET VOTable with %s lines'%len(response.content.split('/n')))
    return response.content

def get_votable(query):
    """ Get the VOTable from NOAO. """
    data = request_votable(query)
    
    fileobj = BytesIO()
    fileobj.write(data)
    votable = vot.parse_single_table(fileobj)
    logging.info("Parsed VOTable with %i rows"%len(votable.array))

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
    if not len(data): return np.array([],dtype=int)
    col = 'original_file'
    return filename2expnum(data[col].data)

def create_nite_safe(data):
    """Convert 'date_obs' to nite. This is safer (using one
    'date2nite' converter) but slower option."""
    col = 'date_obs'
    return date2nite(data[col])

def create_nite_fast(data):
    """Convert 'start_date' to 'nite'. This is the faster option since
    it relies on NOAO to calculate the nite as 'start_date'."""
    if not len(data): return np.array([],dtype='S8')
    col = 'start_date'
    dtype ='S%i'%(len(max(data[col], key=len)))
    nite = data[col].data.astype(dtype)
    nite = np.char.replace(nite,'-','')
    return nite

create_nite = create_nite_fast

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


def download_exposure(expnum,outfile=None,votable=None,certificate=None):
    if certificate is None: certificate=NOAO_CERT 
        
    if np.in1d(expnum,BLACKLIST).sum():
        msg = "Blacklisted exposure: %i"%expnum
        raise Exception(msg)

    data = match_expnum(expnum,votable)
    path = data['reference']
    origsize = data['filesize']

    # If the release date is after today, then must be restricted
    release = dateparse(data['release_date'])
    today = datetime.today()
    if release > today:
        logging.info("Future release date; attempting secure download")
        path = path.replace('http:','https:').replace(':7003',':7503')

    if certificate:
        mtime = datetime.fromtimestamp(os.path.getmtime(certificate))
        msg = "Using certificate: %s"%os.path.basename(certificate)
        msg += " (%s hours old)"%(str(today-mtime)[:-13])
        logging.info(msg)

    if not outfile:
        outfile = 'DECam_{expnum:08d}.fits.fz'.format(expnum=expnum)

    tool = 'curl'
    if tool == 'wget':
        cert = '--certificate %s'%certificate if certificate else ''
        #cert = '--certificate %s'%(certificate if certificate else NOAO_CERT)
        cmd = NOAO_WGET
    elif tool == 'curl':
        cert = '--cert %s'%certificate if certificate else ''
        #cert = '--cert %s'%(certificate if certificate else NOAO_CERT)
        cmd = NOAO_CURL
    cmd = cmd.format(url=path,outfile=outfile,certificate=cert)

    logging.info(cmd)
    retry(cmd,retry=3)
             
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
    
