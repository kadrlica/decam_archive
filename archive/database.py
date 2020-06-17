#!/usr/bin/env python
"""Simple interface interface to a postgres database taking
connection infromation from '.desservices.ini' file.

For more documentation on desservices, see here:
https://opensource.ncsa.illinois.edu/confluence/x/lwCsAw
"""

import os
import tempfile
from collections import OrderedDict as odict
import datetime

import psycopg2
import numpy as np
import pandas as pd
import logging

def desc2dtype(desc):
    """ Covert from postgres type description to numpy dtype.
    Tries to conform to the type mapping on the psycopg2 documentation: 
    https://pythonhosted.org/psycopg2/usage.html"""
    # A list of all string types can be found in:
    # from psycopg2.extensions import string_types
    dtype = []
    for d in desc:
        name = d.name
        code = d.type_code
        size = d.internal_size
        # Booleans
        if code == psycopg2.extensions.BOOLEAN:
            dt = (name, bool)
        # Numeric types
        elif code == psycopg2.extensions.LONGINTEGER:
            dt = (name, long)
        elif code == psycopg2.extensions.INTEGER:
            dt = (name, 'i%i'%size)
        elif code == psycopg2.extensions.FLOAT:
            dt = (name, 'f%i'%size)
        elif code == psycopg2.NUMBER:
            # Catchall for other numbers
            dt = (name, float)
        # Character strings
        elif code == psycopg2.STRING:
            if size > 0:
                dt = (name, 'S%i'%size)
            else:
                # These are TEXT objects of undefined length
                dt = (name, object)
        elif code == psycopg2.extensions.UNICODE:
            # Probably doesn't get called because STRING is a catchall
            dt = (name, 'U%i'%size)
        # Dates and times (should eventually move to np.datetime64)
        elif code == psycopg2.extensions.DATE:
            dt = (name, datetime.date)
        elif code == psycopg2.extensions.TIME:
            dt = (name, datetime.time)
        elif code == psycopg2.extensions.INTERVAL:
            dt = (name, datetime.timedelta)
        elif code == psycopg2.DATETIME:
            dt = (name, datetime.datetime)
        elif code == psycopg2._psycopg.UNKNOWN:
            dt = (name, object)
        # Binary stuff
        elif code == psycopg2.BINARY:
            dt = (name, bytearray)
        elif code == psycopg2.ROWID:
            dt = (name, bytearray)
        else: # Ignore other types for now.
            msg = "Unrecognized type code: "+str(d)
            raise TypeError(msg)
        dtype.append(dt)
    return dtype

def get_desservices(filename=None, section='db-fnal'):
    #if not filename: filename=os.getenv("DES_SERVICES")
    if os.path.exists(".desservices.ini"):
        filename=os.path.expandvars("$PWD/.desservices.ini")
    else:
        filename=os.path.expandvars("$HOME/.desservices.ini")
    logging.debug('Connecting with credentials in %s'%filename)
 
    # ConfigParser throws "no section error" if file does not exist...
    # That's confusing, so 'open' to get a more understandable error
    open(filename) 

    import ConfigParser
    c = ConfigParser.RawConfigParser()
    c.read(filename)
 
    d={}
    d['host']     = c.get(section,'server')
    d['dbname']   = c.get(section,'name')
    d['user']     = c.get(section,'user')
    d['password'] = c.get(section,'passwd')
    d['port']     = c.get(section,'port')
    return d
            
class Database(object):
    def __init__(self,dbname='db-fnal'):
        self.dbname = dbname
        self.conninfo = self.parse_config(section=dbname)
        self.connection = None
        self.cursor = None

    def __str__(self):
        ret = str(self.connection)
        return ret

    def parse_config(self, filename=None, section='db-fnal'):
        return get_desservices(filename,section)

    def connect(self):
        self.connection = psycopg2.connect(**self.conninfo)
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()

    def commit(self):
        self.connection.commit()

    def reset(self):
        self.connection.reset()

    def execute(self,query):
        try: 
            self.cursor.execute(query)
        except Exception as e:
            self.reset()
            raise(e)

    def select(self,query):
        self.execute(query)
        try: 
            return self.cursor.fetchall()
        except Exception as e:
            self.reset()
            raise(e)
    #    self.execute(query, values)

    def load_data(self, table, data, option=None):
        """Load a numpy.recarray or pandas.DataFrame into a table.

        Parameters:
        -----------
        table    : The name of the table to load.
        data     : The data object to load.
        """
        if not isinstance(data,pd.DataFrame):
            data = pd.DataFrame(data)
        
        tmp = tempfile.NamedTemporaryFile(suffix='.csv')
        data.to_csv(tmp.name,index=False)
        logging.debug("Creating temporary file: %s"%tmp.name)
        if option is None: option = ''
        params = dict(table=table,columns=','.join(data.columns),
                      option=option)
        query = "COPY %(table)s (%(columns)s) FROM STDIN WITH CSV HEADER %(option)s;"%params
        logging.debug(query)

        try:
            self.cursor.copy_expert(query,tmp)
        except psycopg2.DataError as e:
            print(e)
            import pdb; pdb.set_trace()
            raise(e)
        del tmp


    def create_table_query(self, **kwargs):
        table = kwargs['table']
        columns = kwargs['columns']

        query = "CREATE TABLE %s ("%table
        #query += ''.join([])

        primary_key = None
        for k,v in sorted(columns.items()):
            ptype = v.get('type','')
            index = v.get('index',None)

            query += '\n%s %s,'%(k,ptype) 

            if not index: continue
            if index == 'PK':
                if primary_key is not None:
                    msg = "Multiple values for PRIMARY KEY"
                    raise ValueError(msg)
                primary_key = 'PRIMARY KEY(%s)\n'%k
    
        if primary_key:
            query += primary_key
        else:
            query = query.strip(',')

        query += ');'
        return query

    def create_index_query(self,**kwargs):
        table = kwargs.get('table')
        columns = kwargs.get('columns')
        indexes = kwargs.get('indexes',None)

        template = "CREATE INDEX {name} ON {table} USING {method} ({column});"
        query = []
        for k,v in sorted(columns.items()):
            method = v.get('index',None)
            if method is None: continue
            if method == 'PK': continue
            name = "%s_%s_idx"%(table.lower(),k.lower())
            params = dict(name=name,table=table,method=method,column=k)
            q = template.format(**params)
            query.append(q)

        if indexes is not None:
            for k,v in sorted(indexes.items()):
                params = dict(table=table,name=k)
                q = v['query'].format(**params)
                query.append(q)

        return query

    def create_indexes(self, **kwargs):
        indexes = self.create_index_query(**kwargs)
        for index in indexes:
            logging.debug(index)
            try:
                self.execute(index)
            except psycopg2.ProgrammingError as e:
                logging.warn(str(e))

    def drop_index_query(self,**kwargs):
        table = kwargs.get('table')
        columns = kwargs.get('columns')
        indexes = kwargs.get('indexes',None)

        template = "DROP INDEX IF EXISTS {name};"
        query = []
        for k,v in sorted(columns.items()):
            method = v.get('index',None)
            if method is None: continue
            if method == 'PK': continue
            name = "%s_%s_idx"%(table.lower(),k.lower())
            q = template.format(name=name)
            query.append(q)

        if indexes is not None:
            for k,v in sorted(indexes.items()):
                if 'PRIMARY KEY' in v: continue
                name = "%s_%s_idx"%(table.lower(),k.lower())
                q = template.format(name=name)
                query.append(q)

        return query

    def drop_indexes(self, **kwargs):
        queries = self.drop_index_query(**kwargs)
        for query in queries:
            logging.debug(query)
            self.execute(query)
            
    def create_table(self,query=None,**kwargs):
        if not query:
            query = self.create_table_query(**kwargs)
            
        logging.debug(query)
        self.execute(query)
        self.commit()

    def table_exists(self,tablename):
        query = """
        SELECT EXISTS (
        SELECT 1
        FROM   information_schema.tables 
        WHERE  table_schema = 'public'
        AND    table_name = '%s'
        );"""%tablename
        return np.all(self.select(query))

    def drop_table(self,tablename):
        query = "DROP TABLE IF EXISTS %s;"%tablename
        return self.execute(query)

    def get_description(self,query=None):
        if query: self.select(query)
        return self.cursor.description

    def get_columns(self,query=None):
        return [d[0] for d in self.get_description(query)]

    def get_dtypes(self,query=None):
        desc = self.get_description(query)
        return desc2dtype(desc)

    def query2rec(self,query):
        # Doesn't work for all data types
        data = self.select(query)
        names = self.get_columns()
        dtypes = self.get_dtypes()
        if not len(data):
            msg = "No data returned by query"
            logging.warn(msg)
            #raise ValueError(msg)
            return np.recarray(0,dtype=dtypes)
        else:
            #return np.rec.array(data,names=names)
            return np.rec.array(data,dtype=dtypes)

    query2recarray = query2rec

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    opts = parser.parse_args()

    db = Database()
    db.connect()
    print db
