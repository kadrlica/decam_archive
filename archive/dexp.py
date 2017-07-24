#!/usr/bin/env python
"""
Module for dealing with postgres database tables.
"""
__author__ = "Alex Drlica-Wagner"
import os
from os.path import dirname, abspath
import logging
import tempfile
from collections import OrderedDict as odict
import copy
import getpass

import numpy as np
import pandas as pd
import yaml
import dateutil.parser
import datetime
import fitsio
import healpy
import numpy.lib.recfunctions as recfn

from astropy.coordinates import Angle, SkyCoord
from astropy.io.fits import Header
import astropy.units as u

from archive.database import Database
from archive.utils import date2nite, get_datadir, ang2pix
import archive.local

def hms2deg(angle):
    return Angle(angle,unit='hourangle').deg

def dms2deg(angle):
    return Angle(angle,unit=u.deg).deg

def cel2gal(ra,dec):
    gal = SkyCoord(ra,dec,unit=u.deg,frame='icrs').galactic
    return gal.l.value, gal.b.value

class DEXP(Database):
    """ SISPI database access. """

    def __init__(self):
        super(DEXP,self).__init__(dbname='db-bliss')
        self.connect()

    def get_date(self,expnum):
        """Get date information for specified exposure number(s). Uses
        SQL to convert SISPI 'date' to a 'nite' string.
        
        Parameters:
        -----------
        expnum : Exposure number(s) to get nite(s) for.

        Returns:
        --------
        date   : Array containing [expnum,date,nite]
        """
        #query = """SELECT id as expnum, date,
        #TO_CHAR(date - '12 hours'::INTERVAL, 'YYYYMMDD')::INTEGER AS nite
        #FROM exposure WHERE id in (%s)"""
        query = """SELECT id as expnum, date,
        TO_CHAR(date - '12 hours'::INTERVAL, 'YYYYMMDD') AS nite
        FROM (SELECT UNNEST('{%s}'::int[]) as id) tmp
        INNER JOIN exposure USING (id)"""

        scalar = np.isscalar(expnum)
        expnum = np.atleast_1d(expnum)
        data = self.query2recarray(query%(','.join(np.char.mod('%d',expnum))))
        
        bad = (expnum != data['expnum'])
        if np.any(bad):
            msg = "'expnum' values don't match"
            msg += '\n'+str(expnum[bad])+'\n'+str(data['expnum'][bad])
            raise ValueError(msg)

        if scalar:
            return data[0]
        return data

    def get_nite(self,expnum):
        """Get the nite(s) for specified exposure number(s). Uses
        utils.date2nite to do the conversion, but checks against SQL
        value derived from SISPI.

        Paramaters:
        -----------
        expnum : Exposure number(s) to get nite(s) for.
        
        Returns: 
        --------
        nite   : Nite(s) corresponding to exposure number(s)
        """
        scalar = np.isscalar(expnum)
        np.atleast_1d(expnum)

        data = self.get_date(expnum)
        nite = date2nite(data['date'])

        np.testing.assert_array_equal(nite,data['nite'],
                                      "'nite' values don't match.")
        
        if scalar:
            return np.asscalar(nite)
        return nite
        
    def expnum2nite(self, expnum):
        return self.get_nite(expnum)

    def load_exposure(self, expnum):
        pass

    def load_exposures(self, expnums):
        pass

class BLISS(DEXP): pass


class Table(object):
    """Base class for postgres table objects."""
    _filename  = os.path.join(get_datadir(),'tables.yaml')
    _section   = None
    
    def __init__(self,config=None,section=None):
        if config is None: config = self._filename
        if section is None: section = self._section
        self.db = DEXP()
        self.load_config(config,section)

    @classmethod
    def get_tablename(cls, config=None):
        if config is None:
            config = yaml.load(open(cls._filename,'r'))
        return config[cls._section]['table']

    def load_config(self,config, section=None):
        if config is None: return config

        if isinstance(config,basestring):
            logging.debug("Loading configuration file: %s..."%config)
            config = yaml.load(open(config,'r'))
        elif isinstance(config,dict):
            config = copy.deepcopy(config)
        else:
            msg = "Unrecognized type for table configuration: %s"
            raise TypeError(msg)

        if section is None:
            self.config = config
        else:
            self.config = config[section]
        self.tablename = self.config['table']

        # Check the config
        self.check_config()

        return config

    def check_config(self):
        assert 'columns' in self.config

        # Check that the columns match
        if self.exists():
            cfgcol = sorted(map(str.upper,self.config['columns'].keys()))
            dbcol = sorted(map(str.upper,self.get_columns()))
            if not np.all(cfgcol==dbcol):
                msg = "Config columns do not match database."
                raise ValueError(msg)

    def exists(self):
        return self.db.table_exists(self.tablename)
        
    def get_columns(self):
        query = "select * from %s limit 0;"%self.tablename
        return self.db.get_columns(query)

    def create_table(self):
        return self.db.create_table(**self.config)

    def drop_table(self):
        self.db.drop_table(self.tablename)

    def grant_table(self):
        query = "grant select on %s to public;"%self.tablename
        self.db.execute(query)

    def create_indexes(self):
        self.db.create_indexes(**self.config)

    def drop_indexes(self):
        self.db.drop_indexes(**self.config)
        
    def build_table(self,force=True):
        if force: self.drop_table()
        self.create_table()
        self.create_indexes()
        self.grant_table()

    def load_table(self,data,option=None):
        self.db.load_data(self.tablename,data,option)

    def load_chunks(self,data,chunk_size=1000):
        if np.isscalar(data): data = [data]
        nchunks = len(data)//chunk_size + 1
        opts = np.get_printoptions()
        np.set_printoptions(threshold=3,edgeitems=1)
        for i,chunk in enumerate(np.array_split(data,nchunks)):
            first,last = chunk[0],chunk[-1]
            if isinstance(chunk[0],str): 
                first,last = map(os.path.basename,[first,last])
            msg = "(%i/%i) Loading chunk of %i rows\n"%(i+1,nchunks,len(chunk))
            msg+= "  [%s - %s]..."%(first,last)
            logging.info(msg)
            self.load_table(chunk)
        np.set_printoptions(**opts)

    def get_description(self):
        return self.db.get_description("select * from %s limit 0;"%self.tablename)

    def get_dtypes(self):
        return self.db.get_dtypes("select * from %s limit 0;"%self.tablename)

    def get_expnum(self):
        query = "select expnum from %s;"%self.tablename
        return self.db.query2rec(query)['expnum']
    
FILTER_DICT = odict([
        ("u DECam c0006 3500.0 1000.0",'u'),
        ("g DECam SDSS c0001 4720.0 1520.0",'g'),
        ("r DECam SDSS c0002 6415.0 1480.0",'r'),
        ("i DECam SDSS c0003 7835.0 1470.0",'i'),
        ("z DECam SDSS c0004 9260.0 1520.0",'z'),
        ("Y DECam c0005 10095.0 1130.0",'Y'),
        ("VR DECam c0007 6300.0 2600.0",'VR'),
        ("N964 DECam c0008 9645.0 94.0","N964"),
        ("solid plate 0.0 0.0",'block'),
        ("pinhole plate 0.0 0.0","pin"),
        ("Empty 0.0 0.0","empty"),
        ])

class ExposureTable(Table):
    """Object for managing the exposure table."""
    _filename  = os.path.join(get_datadir(),'tables.yaml')
    _section   = 'exposure'

    def __init__(self):
        super(ExposureTable,self).__init__(self._filename,self._section)
    
    #def check_config(self):
    #    """Check that the configuration is valid."""
    #    pass

    #def load_exposures(self, expnums, chunk_size=1000):
    #    """Load a list of exposure numbers.
    #    
    #    Parameters:
    #    -----------
    #    expnums    : List of exposure numbers.
    #    chunk_size : Number of exposures to process and upload at a time.
    # 
    #    Returns:
    #    --------
    #    None
    #    """
    #    if np.isscalar(expnums): expnums = [expnums]
    #    nchunks = len(expnums)//chunk_size + 1
    #    opts = np.get_printoptions()
    #    np.set_printoptions(threshold=3,edgeitems=1)
    #    for i,chunk in enumerate(np.array_split(expnums,nchunks)):
    #        first = os.path.basename(chunk[0])
    #        last = os.path.basename(chunk[-1])
    #        msg = "(%i/%i) Loading chunk [%s - %s]..."%(i+1,nchunks,first,last)
    #        logging.info(msg)
    #        self.load_exposure_chunk(chunk)
    #    np.set_printoptions(**opts)

    def load_table(self, data):
        """Load a chunk of exposures.
        
        Parameters:
        -----------
        data : Set of expnums to load.
        
        Returns:
        --------
        None
        """
        expnum = np.atleast_1d(data)
        df = pd.DataFrame(self.read_headers(expnum))
        self.load_table(df)

    def read_headers(self, expnums, keys=None, multiproc=True):
        """Read the exposure information out of the file headers.
        
        Parameters:
        -----------
        expnums   : The list of exposure numbers to process.
        keys      : The list of columns to pull out (defaults to config)
        multiproc : Flag for using multiprocessing

        Returns:
        --------
        df : A pandas.DataFrame containing the header information.
        """
        if keys is None: keys = self.config['columns'].keys()

        headers = archive.local.read_exposure_headers(expnums, multiproc=multiproc)
        hdrs = [self.augment_header(h) for h in headers]
        return pd.DataFrame.from_records(hdrs,columns=keys)

    @classmethod
    def augment_header(cls, header):
        """ Augment existing file header information with additional
        exposure information.

        Parameters:
        -----------
        header : The original file header

        Returns:
        --------
        hdr : The dictionary of exposure information
        """
        hdr = dict([(k,header.get(k)) for k in header.keys()])
        for k,v in hdr.items():
            if isinstance(v,basestring):
                hdr[k] = v.strip()

        if hdr.get('RA'):     hdr['RADEG'] = hms2deg(hdr['RA'])
        if hdr.get('DEC'):    hdr['DECDEG'] = dms2deg(hdr['DEC'])
        if hdr.get('TELRA'):  hdr['TRADEG'] = hms2deg(hdr['TELRA'])
        if hdr.get('TELDEC'): hdr['TDECDEG'] = dms2deg(hdr['TELDEC'])
        if hdr.get('FILTER'): hdr['BAND'] = FILTER_DICT.get(hdr['FILTER'])
        if 'SEQID' in hdr:  hdr['FIELD'] = hdr.get('SEQID').split()[0][:30]
        if 'DATE-OBS' in hdr:  hdr['NITE'] = date2nite(hdr['DATE-OBS'])

        hdr['DATE_OBS']    = hdr.get('DATE-OBS')
        hdr['ALTITUDE']    = hdr.get('OBS-ELEV')
        hdr['LONGITUDE']   = hdr.get('OBS-LONG')
        hdr['LATITUDE']    = hdr.get('OBS-LAT')
        hdr['MJD_OBS']     = hdr.get('MJD-OBS')
        hdr['TIME_OBS']    = hdr.get('TIME-OBS')
        hdr['OBSERVATORY'] = hdr.get('OBSERVAT')
        hdr['INSTRUMENT']  = hdr.get('INSTRUME')
        hdr['TELESCOPE']   = hdr.get('TELESCOP')
        hdr['TILING']      = hdr.get('TILING')

        # Can't have NaNs in int column
        try: hdr['HPIX'] = int(ang2pix(4096,hdr['RADEG'],hdr['DECDEG']))
        except KeyError: hdr['HPIX'] = -1

        # Convert to Galactic coordinates
        glon,glat = cel2gal(hdr['RADEG'],hdr['DECDEG'])
        hdr['GLON'] = glon
        hdr['GLAT'] = glat

        #hdr['FILEPATH'] = archive.local.get_path(hdr['EXPNUM'],hdr['NITE'])
        hdr['FILETYPE'] = 'raw'
         
        return hdr

    #def get_expnum(self):
    #    query = 'select expnum from %s;'%self.tablename
    #    return self.db.query2recarray(query)['expnum']
    #    #try:
    #    #    expnum = self.db.query2recarray(query)['expnum']
    #    #except ValueError:
    #    #    #expnum = np.rec.recarray(0,dtype=[('expnum',int)])
    #    #    expnum = np.array([],dtype=int)
    #    #
    #    #return expnum

    def delete_expnum(self,expnum):
        expnum = np.atleast_1d(expnum)
        expstr = ','.join('%s'%e for e in expnum)
        query = 'delete from %s where expnum in (%s);'%(self.tablename,expstr)
        logging.debug(query)
        self.db.execute(query)

class ImageTable(Table):
    """Object for managing the 'image' table."""
    _filename  = os.path.join(get_datadir(),'tables.yaml')
    _section   = 'image'

    def __init__(self):
        super(ImageTable,self).__init__(self._filename,self._section)
    
    #def check_config(self):
    #    """Check that the configuration is valid."""
    #    pass
    
    #def load_images(self, filepaths, chunk_size=1000):
    #    """Load a list of images.
    #    
    #    Parameters:
    #    -----------
    #    expnums    : List of exposure numbers.
    #    chunk_size : Number of exposures to process and upload at a time.
    # 
    #    Returns:
    #    --------
    #    None
    #    """
    #    if np.isscalar(filepaths): filepaths = [filepaths]
    #    nchunks = len(filepaths)//chunk_size + 1
    #    opts = np.get_printoptions()
    #    np.set_printoptions(threshold=3,edgeitems=1)
    #    for i,chunk in enumerate(np.array_split(filepaths,nchunks)):
    #        first = os.path.basename(chunk[0])
    #        last = os.path.basename(chunk[-1])
    #        msg = "(%i/%i) Loading chunk [%s - %s]..."%(i+1,nchunks,first,last)
    #        logging.info(msg)
    #        self.load_image_chunk(chunk)
    #    np.set_printoptions(**opts)

    def load_table(self, data):
        """Load a chunk of images
        
        Parameters:
        -----------
        data : The list of filepaths to load.
        
        Returns:
        --------
        None
        """
        filepaths = np.atleast_1d(data)
        df = pd.DataFrame(self.read_headers(filepaths))
        super(ImageTable,self).load_table(df)
        
    def read_headers(self, filepaths, keys=None, multiproc=True):
        """Read the image information out of the file headers.
        
        Parameters:
        -----------
        expnums   : The list of exposure numbers to process.
        keys      : The list of columns to pull out (defaults to config)
        multiproc : Flag for using multiprocessing

        Returns:
        --------
        df : A pandas.DataFrame containing the header information.
        """
        if keys is None: keys = self.config['columns'].keys()

        headers = archive.local.read_image_headers(filepaths, multiproc=multiproc)
        hdrs = [self.augment_header(h) for h in headers]
        return pd.DataFrame.from_records(hdrs,columns=keys)

    @classmethod
    def augment_header(cls, header):
        """ Augment existing file header information with additional
        image information.

        Parameters:
        -----------
        header : The original file header

        Returns:
        --------
        hdr : The dictionary of exposure information
        """
        hdr = dict([(k,header.get(k)) for k in header.keys()])
        for k,v in hdr.items():
            if isinstance(v,basestring):
                hdr[k] = v.strip()
        
        if hdr.get('AIRMASS') is None: 
            hdr['AIRMASS'] = 1/np.cos(np.radians(hdr.get('ZD',np.nan)))

        hdr['CCDNAME']  = hdr.get('DETPOS')
        hdr['FILETYPE'] = 'immask'

        # Can't have NaNs in int column
        try: hdr['HPIX'] = int(ang2pix(4096,hdr['RA_CENT'],hdr['DEC_CENT']))
        except KeyError: hdr['HPIX'] = -1
         
        return hdr

    def delete_image(self,filename):
        filename = np.atleast_1d(filename)
        filestr = ','.join('%s'%f for f in filename)
        query = 'delete from %s where filename in (%s);'%(self.tablename,filestr)
        logging.debug(query)
        self.db.execute(query)

class CatalogTable(Table):
    """Object for managing the objects table."""
    _filename  = os.path.join(get_datadir(),'tables.yaml')
    _section   = 'catalog'

    def __init__(self):
        super(CatalogTable,self).__init__(self._filename,self._section)
    
class ArchiveTable(Table):
    _filename  = os.path.join(get_datadir(),'tables.yaml')
    _section   = 'archive'

    #def load_archive_info(self, filenames, chunk_size=1e5):
    #    """Load file archive info.
    #    
    #    Parameters:
    #    -----------
    #    filenames : SE files to archive.
    #    chunk_size : Number of files to process and upload at once.
    #    
    #    Returns:
    #    --------
    #    None
    #    """
    #    filenames = np.atleast_1d(filenames)
    #    nchunks = len(filenames)//chunk_size + 1
    #    opts = np.get_printoptions()
    #    np.set_printoptions(threshold=3,edgeitems=1)
    #    for i,chunk in enumerate(np.array_split(filenames,nchunks)):
    #        first = os.path.basename(chunk[0])
    #        last = os.path.basename(chunk[-1])
    #        msg = "(%i/%i) Loading chunk [%s - %s]..."%(i+1,nchunks,first,last)
    #        logging.debug(msg)
    #        data = self.create_archive(chunk)
    #        self.load_table(data,option='FORCE NOT NULL compression')
    #    np.set_printoptions(**opts)

    def load_table(self, data):
        """Load a chunk of images
        
        Parameters:
        -----------
        data : The list of filepaths to load.
        
        Returns:
        --------
        None
        """
        filepaths = np.atleast_1d(data)
        data = self.create_archive(filepaths)
        option='FORCE NOT NULL compression'
        super(ArchiveTable,self).load_table(data,option=option)

    def create_view(self):
        """ Create a view of the table with an extra filepath column. """
        query = "create view %s as (select *,path||'/'||filename||compression as filepath from %s);"%(self.tablename,'_'+self.tablename)
        self.db.execute(query)

    def create_archive(self, filenames):
        dtype = self.get_dtypes()
        arc = np.recarray(len(filenames),dtype=dtype)
        parsed = archive.local.parse_reduced_file(filenames)

        for n in parsed.dtype.names:
            arc[n.lower()] = parsed[n.lower()]

        arc['status'] = 'processed'
        return arc

    def get_filename(self):
        """Get the filename with compression."""
        query = 'select filename||compression as filename from %s;'%self.tablename
        return self.db.query2recarray(query)['filename'].astype(str)

    def get_filepath(self,condition=None):
        """Get the filepath from the path, filename, and compression."""
        if condition is None: condition = ''
        query = "select path||'/'||filename||compression as filepath from %s %s;"%(self.tablename,condition)
        return self.db.query2recarray(query)['filepath'].astype(str)

    def delete_filename(self,filename):
        filename = np.atleast_1d(filename)
        filename = map(os.path.basename,filename)
        filestr = ','.join("'%s'"%f for f in filename)
        query = "delete from %s where filename in (%s);"%(self.tablename,filestr)
        logging.debug(query)
        self.db.execute(query)

    def set_status(self,status,filename):
        filename = np.atleast_1d(filename)
        values = ','.join(self.db.cursor.mogrify("'%s'"%f) for f in filename)
        query = "update %s set status = '%s' where filename in (%s);"%(self.tablename,status,values)
        logging.debug(query)
        self.db.execute(query)

class ObjectsTable(Table):
    """Object for managing the objects table."""
    _filename  = os.path.join(get_datadir(),'tables.yaml')
    _section   = 'objects'

    _hdrhdu = 'LDAC_IMHEAD'
    _objhdu = 'LDAC_OBJECTS'

    def __init__(self):
        super(ObjectsTable,self).__init__(self._filename,self._section)
        self.archive = ArchiveTable()

    def check_config(self):
        """Check that the configuration is valid."""
        pass

    def load_catalogs(self, filepaths, chunk_size=1000, force=False):
        """Load a list of catalogs.
        
        Parameters:
        -----------
        catalogs   : List of catalogs (files or arrays)
        chunk_size : Number of catalogs to process and upload at once.

        Returns:
        --------
        None
        """
        filepaths = np.atleast_1d(filepaths)
        for i,filepath in enumerate(filepaths):
            msg = "(%i/%i) Loading %s..."%(i+1,len(filepaths),os.path.basename(filepath))
            logging.info(msg)
            self.load_catalog(filepath,force)

    def load_catalog(self, filepath, force=False):
        """Load an object catalog.
        
        Parameters:
        -----------
        filepath : Full path to the file to upload.
        
        Returns:
        --------
        None
        """
        arc = ArchiveTable()
        parsed = archive.local.parse_reduced_file(filepath)
        filename = parsed['filename']
        query = "select status from %s where filename = '%s'"%(arc.tablename,filename)
        status = arc.db.query2recarray(query)['status']

        if len(status) == 0:
            arc.load_archive_info(filepath)
        elif status[0] == 'uploaded' and not force:
            msg = "Filename already uploaded: %s"%filename
            logging.warn(msg)
            # raise exception?
            return

        try:
            data = pd.DataFrame(self.create_catalog(filepath))
            self.db.load_data(self.tablename,data)
            logging.info("  Loaded %i objects."%len(data))
            stat = 'uploaded'
        except Exception as e:
            stat = 'failed'
            msg = str(e)
            msg += "\nFailed to upload %s."%filepath
            logging.warn(msg)
        ArchiveTable().set_status(stat,filename)

    def delete_catalog(self, filepath):
        filepath = np.atleast_1d(filepath)
        filename = map(os.path.basename,filepath)
        filestr = ','.join("'%s'"%f for f in filename)
        query = "delete from %s where filename in (%s);"%(self.tablename,filestr)
        logging.debug(query)
        self.db.execute(query)

    @classmethod
    def parse_catalog(cls, catalog):
        if isinstance(catalog,basestring):
            return fitsio.FITS(catalog)
        elif isinstance(catalog,fitsio.FITS):
            return catalog
        else:
            msg = "Unrecognized catalog type: '%s'"%type(catalog)
            raise Exception(msg)

    def create_catalog(self, catalog):
        """ Augment existing catalog information with additional
        exposure information.

        Parameters:
        -----------
        catalog : The original catalog file

        Returns:
        --------
        cat : The catalog recarray
        """
        # This is a pretty messy and hardcoded function...
        fits = self.parse_catalog(catalog)
        hdr = odict(self.create_header(fits))
        data = fits[self._objhdu].read()

        # The names of the columns in the database
        names = self.db.get_columns('select * from %s limit 0;'%self.tablename)
        names = sorted(map(str.upper,names))

        data_dtype = [d for d in data.dtype.descr if d[0] in names]

        hdr_dtype = [('EXPNUM','>i4'),('CCDNUM','>i2'),
                     ('NITE','>i4'),('BAND','S5')]

        extra_dtype = [('FILENAME','S60'),('REQNUM','>i2'),('ATTNUM','>i2')]
        extra_dtype += [('OBJECT_NUMBER',data.dtype['NUMBER']),
                        ('RA',data.dtype['ALPHAWIN_J2000']),
                        ('DEC',data.dtype['DELTAWIN_J2000'])]

        aper_dtype = []
        for i in range(data['FLUX_APER'].shape[-1]):
            aper_dtype += [('FLUX_APER_%i'%(i+1),'>f4')]
            aper_dtype += [('FLUXERR_APER_%i'%(i+1),'>f4')]

        dtype = sorted(data_dtype+hdr_dtype+aper_dtype+extra_dtype)
        cat = np.recarray(data.shape,dtype=dtype)

        if not np.all(np.in1d(names,cat.dtype.names)):
            msg = "Missing columns in catalog:\n"
            msg += str(np.array(names)[~np.in1d(names,cat.dtype.names)])
            raise Exception(msg)

        if not np.all(np.in1d(cat.dtype.names,names)):
            msg = "Extra columns in catalog:\n"
            msg += str(np.array(cat.dtype.names)[~np.in1d(cat.dtype.names,names)])
            raise Exception(msg)
            
        for name,dt in data_dtype:
            cat[name] = data[name]

        cat['OBJECT_NUMBER'] = data['NUMBER']
        cat['RA']  = data['ALPHAWIN_J2000']
        cat['DEC'] = data['DELTAWIN_J2000']

        for i in range(data['FLUX_APER'].shape[-1]):
            cat['FLUX_APER_%i'%(i+1)] = data['FLUX_APER'][:,i]
            cat['FLUXERR_APER_%i'%(i+1)] = data['FLUXERR_APER'][:,i]

        for name,dt in hdr_dtype:
            cat[name] = hdr[name]

        filename = os.path.basename(fits._filename)
        reqnum,attnum = map(int,filename.split('_')[3].strip('r').split('p'))
        cat['FILENAME'] = filename
        cat['REQNUM'] = reqnum
        cat['ATTNUM'] = attnum

        return cat

    @classmethod
    def create_header(cls, catalog):
        if isinstance(catalog,np.ndarray):
            data = catalog
        else:
            fits = cls.parse_catalog(catalog)
            data = fits[cls._hdrhdu].read()[0][0]
        data = data[~np.char.startswith(data,'        =')]
        s = '\n'.join(data)
        return Header.fromstring(s,sep='\n')

class JobsTable(Table):
    """Object for managing the jobs table."""
    _filename  = None
    _section   = None
    
    def __init__(self):
        super(JobsTable,self).__init__(self._filename,self._section)
        self.tablename = 'jobs'

    def get_expnum(self):
        query = "select jobid::INT as expnum from %s;"%(self.tablename)
        return self.db.query2rec(query)['expnum']

    def get_expnum_done(self):
        query = "select jobid::INT as expnum from %s where status='done';"%(self.tablename)
        return self.db.query2rec(query)['expnum']
        

class ZeropointTable(Table):
    """Object for managing the zeropoint table."""
    _filename  = os.path.join(get_datadir(),'tables.yaml')
    _section   = 'zeropoint'
    
    ## From docdb:9750
    #KTERMS = odict([
    #        ('u',0.436),
    #        ('g',0.192),
    #        ('r',0.097),
    #        ('i',0.071),
    #        ('z',0.083),
    #        ('Y',0.067),
    #        ])
            
    #def load_zeropoints(self, filepaths, chunk_size=1000):
    #    """Load a list of catalogs.
    #    
    #    Parameters:
    #    -----------
    #    filepaths  : List of merged zeropoint files to load
    #    chunk_size : Number of files to process and upload at once.
    # 
    #    Returns:
    #    --------
    #    None
    #    """
    #    filepaths = np.atleast_1d(filepaths)
    #    nchunks = len(filepaths)//chunk_size + 1
    #    opts = np.get_printoptions()
    #    np.set_printoptions(threshold=3,edgeitems=1)
    #    for i,chunk in enumerate(np.array_split(filepaths,nchunks)):
    #        first = os.path.basename(chunk[0])
    #        last = os.path.basename(chunk[-1])
    #        msg = "(%i/%i) Loading chunk [%s - %s]..."%(i+1,nchunks,first,last)
    #        logging.info(msg)
    #        data = self.create_zeropoints(chunk)
    #        self.load_table(data)
    #    np.set_printoptions(**opts)

    def load_table(data):
        """Load a chunk of zeropoint files
        
        Parameters:
        -----------
        data : The list of filepaths to load.
        
        Returns:
        --------
        None
        """
        filepaths = np.atleast_1d(data)
        data = self.create_zeropoints(filepaths)
        super(ZeropointTable,self).load_table(data)

    def create_zeropoints(self, filepaths):
        """ Create the zeropoint array for upload.

        Parameters:
        -----------
        filepaths: Merged zeropoint files to parse.

        Returns:
        --------
        zeropoints: Numpy array of zeropoints
        """
        data,filename = [],[]
        for f in filepaths:
            try:
                d = pd.read_csv(f).to_records(index=False)
            except ValueError as e:
                # File doesn't exist?
                msg = str(e) + '\n Skipping %s...'%f
                logging.warn(msg)
                continue
            data += [d]
            filename += len(d)*[os.path.basename(f)]
            
        filename = np.array(filename)
        data = np.hstack(data)
        if len(filename) != len(data):
            msg = "Length mismatch in data and filename"
            raise ValueError(msg)

        dtype = self.get_dtypes()
        zp = np.recarray(len(data),dtype=dtype)
        
        parsed = archive.local.parse_reduced_file(data['FILENAME'])

        # Some sanity checks
        if np.any(parsed['ccdnum'] != data['CCDNUM']):
            msg = "CCDNUMs do not match."
            raise ValueError(msg)
        if np.any(parsed['expnum'] != data['EXPNUM']):
            msg = "EXPNUMs do not match."
            raise ValueError(msg)
        
        zp['filename'] = filename
        zp['catalogname'] = data['FILENAME']
        zp['expnum'] = data['EXPNUM']
        zp['ccdnum'] = data['CCDNUM']
        zp['mag_zero'] = data['NewZP']
        zp['sigma_mag_zero'] = data['NewZPrms']
        zp['flag'] = data['NewZPFlag']
        zp['band'] = parsed['band']
        zp['source'] = 'expCalib'

        ## Calculate the effective zeropoint
        #effzp = self.effective_zeropoints(zp)
        #bad = np.isnan(effzp)
        #effzp[bad] = -999
        #zp['eff_mag_zero'] = effzp
        #zp['flag'][bad] -= int(1e5)

        return zp

    def effective_zeropoints(self,data):
        """ 
        DEPRECATED: ADW 2017-07-23
        The zeropoints from expCalib are not corrected for exposure
        time or airmass.

        Parameters:
        -----------
        data : recarray containing 'expnum' and 'mag_zero' columns.
        
        Returns:
        --------
        effzp : the effective zeropoint
        """
        d = pd.DataFrame(data[['expnum','mag_zero']])
        exp = self.db.query2rec('select expnum, exptime, band, zd, 1/COS(RADIANS(zd)) as airmass from exposure e;')
        exp = pd.DataFrame(exp)
        merge = d.merge(exp,on='expnum')
        kterms = np.array(map(self.KTERMS.get,merge['band']))
        effzp = (merge['mag_zero'] + 2.5*np.log10(merge['exptime'])-merge['airmass']*kterms)
        return effzp.values
                         
    def read_zeropoints(self, filepath):
        pd.read_csv(filepath)

class ProctagTable(Table):
    """Object for managing the proctag table."""
    _filename  = os.path.join(get_datadir(),'tables.yaml')
    _section   = 'proctag'
    _tags      = os.path.join(get_datadir(),'proctag.yaml')

    def __init__(self):
        super(ProctagTable,self).__init__(self._filename,self._section)
        self._load_tags()

    def _load_tags(self):
        self.tags = yaml.load(open(self._tags))

    def load_proctag(self, tag, query=None, expnum=None):
        """ Create the proctag and load to db. Use tag lookup query.

        Parameters:
        -----------
        tag:    tag to create
        query:  explicit query to select expnums
        expnum: explicit list of expnums

        Returns:
        --------
        proctag: array of proctag values
        """
        proctag = self.create_proctag(tag,query,expnum)
        self.db.load_data(self.tablename,proctag)
        return proctag

    def create_proctag(self, tag, query=None, expnum=None):
        """ Create the proctag data array.
        
        Parameters:
        -----------
        tag:     tag to create
        query:   explicit query to select expnums
        expnum:  explicit list of expnums

        Returns:
        --------
        proctag: array of proctag values
        """
        if expnum is not None and query is not None:
            msg = "Cannot specify 'query' and 'expnum'."
            raise Exception(msg)

        if expnum is None:
            if query is None: query = self.tags[tag]['query']
            expnum = self.db.query2rec(query)['expnum']
        else:
            expnum = np.atleast_1d(expnum)
        
        data = np.recarray(len(expnum),dtype=self.get_dtypes())
        data['expnum'] = expnum
        data['tag'] = tag
        data['created_date'] = datetime.datetime.now()
        data['created_by'] = getpass.getuser()
        return data

def expnum2nite(expnum):
    """Standalone call to the nite from DExp

    Paramaters:
    -----------
    expnum : Exposure number(s) to get nite(s) for.
    
    Returns: 
    --------
    nite   : Nite(s) corresponding to exposure number(s)
    """
    pass

def create_table(cls, force=False):
    """Create the postgres table.

    Parameters:
    -----------
    cls:   The table class
    force: Drop and recreate table if it already exists.

    Returns:
    --------
    None
    """
    tab = cls()
    tablename = tab.tablename
    if tab.exists():
        if force:
            tab.drop_table()
        else:
            msg = "The '%s' table already exists."%tablename
            logging.info(msg)
            return
    tab.create_table()
    tab.grant_table()

def finalize_table(cls):
    """Finalize the postgres table by creating indexes and granting access.

    Parameters:
    -----------
    cls:   The table class

    Returns:
    --------
    None
    """
    tab = cls()
    tab.create_indexes()
    tab.grant_table()

#def create_exposure_table(force=False):
#    """Create the postgres exposure table."""
#    return create_table(ExposureTable)
# 
#def create_objects_table(force=False):
#    """Create the postgres se_objects table."""
#    return create_table(ObjectsTable)
# 
#def create_archive_table(force=False):
#    """Create the postgres se_objects table."""
#    return create_table(ArchiveTable)
# 
#def create_zeropoint_table(force=False):
#    """Create the postgres se_objects table."""
#    return create_table(ZeropointTable)

def load_exposure_table(expnum=None,chunk_size=100,multiproc=False,force=False):
    """Standalone function to load the exposure table.
    """
    expnum = np.atleast_1d(expnum)
    if not len(expnum) or expnum[0] is None:
        expnum = np.unique(archive.local.get_inventory()['expnum'])

    tab = ExposureTable()

    if force:
        sel = np.in1d(expnum,tab.get_expnum())
        if np.any(sel): tab.delete_expnum(expnum[sel])

    sel = ~np.in1d(expnum,tab.get_expnum())
    if not np.any(sel):
        msg = "No new exposures to upload"
        raise Exception(msg)

    return tab.load_chunks(expnum[sel],chunk_size)

def load_archive_table(expnum=None,chunk_size=1e3,multiproc=True,force=False):
    """Load the exposure table from the file inventory on disk.

    Parameters:
    -----------
    expnum     : list of exposures to load
    chunk_size : size of chunk of exposures to load
    multiproc  : number of multiprocessing cores to use
    force      : overwrite of existing file info

    Returns:
    --------
    None
    """
    tab = ArchiveTable()

    expnum = np.atleast_1d(expnum)

    # Only get exposures marked as 'done'
    if len(expnum)==0 or expnum[0]==None:
        expnum = JobsTable().get_expnum_done()

    logging.debug("Loading %i exposures..."%len(expnum))

    # The query for filepaths actually takes a long time, so split up
    nchunks = len(expnum)//chunk_size + 1
    for i,chunk in enumerate(np.array_split(expnum,nchunks)):
        msg = "(%i/%i) Getting reduced files from disk..."%(i+1,nchunks)
        logging.info(msg)

        # Get all the files
        files = archive.local.get_reduced_files(expnum=chunk,suffix='_*.fits*',
                                                multiproc=multiproc)
        psffiles = archive.local.get_psfex_files(expnum=chunk,multiproc=multiproc)
        zpfiles = archive.local.get_zeropoint_files(expnum=chunk,multiproc=multiproc)
        filepath = np.unique(np.concatenate([files,zpfiles,psffiles]))

        path,sep,filename = np.char.rpartition(filepath,'/').T
        if force:
            logging.debug("Removing old filenames...")
            sel = np.in1d(filename,tab.get_filename())
            if np.any(sel): tab.delete_filename(filename[sel])
     
        sel = ~np.in1d(filename,tab.get_filename())
        if not np.any(sel):
            logging.warn("No new filenames to upload.")
            continue
     
        logging.debug("Loading archive...")
        tab.load_table(filepath[sel],1e5)

    # Update the band information with a query
    logging.debug("Updating zeropoint bands:")
    query = """update %(archive)s a set band = e.band from %(exposure)s 
               where e.expnum = a.expnum and a.band is NULL
            """%dict(archive=tab.tablename,exposure=ExposureTable().tablename)
    logging.debug(query)
    tab.db.execute(query)

    return

def load_object_table(filepath=None,chunk_size=100,force=False):
    """Standalone function to load the object table.
    """
    obj = ObjectsTable()
    arc = ArchiveTable()

    filepath = np.atleast_1d(filepath)

    logging.debug("Getting catalog filenames from %s..."%arc.tablename)
    query = "select filename,path||'/'||filename||compression as filepath,status from %s where filetype = 'fullcat';"%(arc.tablename)
    logging.debug(query)
    values = arc.db.query2recarray(query)

    # Upload all files that are 'processed'
    if len(filepath)==0 or filepath[0] is None:
        filepath = values['filepath'][values['status']=='processed']

    # Don't upload files that are already uploaded
    if not force:
        filename = archive.local.parse_reduced_file(filepath)['filename']
        sel=np.in1d(filename,values['filename'][values['status']!='uploaded'])
        filepath = filepath[sel]

    if not len(filepath):
        logging.warn("No new files to upload.")
        return

    logging.debug("Loading objects...")
    return obj.load_catalogs(filepath,chunk_size,force)

def load_zeropoint_table(expnum=None,chunk_size=100,multiproc=True,force=False):
    """Standalone function to load the zeropoint table.
    """

    tab = ZeropointTable()

    expnums = np.atleast_1d(expnum)

    # Upload all files that are 'processed'
    logging.debug("Getting zeropoint files...")
    database = True
    if database:
        arc = ArchiveTable()
        query = "select expnum, path||'/'||filename||compression as filepath from %s where filetype='allzp';"%arc.tablename
        inv = arc.db.query2rec(query)
        filepath = inv['filepath']
        if not (len(expnum)==0 or expnum[0]==None):
            filepath = filepath[np.in1d(inv['expnum'],expnums)]
    else:
        filepath = archive.local.get_zeropoint_files(expnums=expnums)

    loaded = tab.db.query2recarray('select filename from %s;'%tab.tablename)

    # Don't upload files that are already uploaded
    if not force:
        filename = np.array(map(os.path.basename,filepath))
        sel = ~np.in1d(filename,loaded['filename'])
        filepath = filepath[sel]

    if not len(filepath):
        logging.warn("No new zeropoints to load")
        return

    logging.debug("Loading zeropoints...")
    return tab.load_chunks(filepath,chunk_size)

def load_image_table(expnum=None,chunk_size=1e4,multiproc=True,force=False):
    """Standalone function to load the image table.
    """
    
    tab = ImageTable()
    expnum = np.atleast_1d(expnum)

    logging.debug("Getting image files...")
    database=True
    if database:
        arc = ArchiveTable()
        query = "select expnum, path||'/'||filename||compression as filepath from %s where filetype='immask';"%arc.tablename
        inv = arc.db.query2rec(query)
        filepath = inv['filepath'].astype(str)
        if not (len(expnum)==0 or expnum[0]==None):
            filepath = filepath[np.in1d(inv['expnum'],expnum)]
        del arc
    else:
        filepath = archive.local.get_immask_files(expnum=expnum)

    loaded = tab.db.query2rec('select filename from %s;'%tab.tablename)

    # Need to implement `force`...

    # Don't upload files that are already uploaded
    if not force:
        filename = np.array(map(os.path.basename,filepath))
        sel = ~np.in1d(filename,loaded['filename'])
        filepath = filepath[sel]

    if not len(filepath):
        logging.warn("No new images to load.")
        return

    logging.debug("Loading images...")
    return tab.load_chunks(filepath,chunk_size)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
