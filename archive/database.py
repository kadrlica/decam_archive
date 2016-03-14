#!/usr/bin/env python
"""Simple interface interface to a postgres database taking
connection infromation from '.desservices.ini' file.

For more documentation on desservices, see here:
https://opensource.ncsa.illinois.edu/confluence/x/lwCsAw

"""

import os
import psycopg2
import numpy as np
import logging

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
        #if not filename: filename=os.getenv("DES_SERVICES")
        if os.path.exists(".desservices.ini"):
            filename=os.path.expandvars("$PWD/.desservices.ini")
        else:
            filename=os.path.expandvars("$HOME/.desservices.ini")
        logging.debug('.desservices.ini: %s'%filename)

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

    def connect(self):
        self.connection = psycopg2.connect(**self.conninfo)
        self.cursor = self.connection.cursor()

    def execute(self,query):
        self.cursor.execute(query)      
        try: 
            return self.cursor.fetchall()
        except Exception, e:
            self.reset()
            raise(e)
        
    def reset(self):
        self.connection.reset()

    def get_columns(self):
        return [d[0] for d in self.cursor.description]

    def query2recarray(self,query):
        # Doesn't work for all data types
        data = self.execute(query)
        names = self.get_columns()
        return np.rec.array(data,names=names)

    def get_nite(self,expnum):
        query = "select date from exposure where id = '%s'"%(expnum)
        data = self.query2recarray(query)
        date = data[0]['date']
        nite = int(date.strftime('%Y%m%d'))
        if date.hour < 13: nite -= 1
        return nite
    
def expnum2nite(expnum):
    db = Database()
    db.connect()
    return db.get_nite(expnum)
    
if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    opts = parser.parse_args()

    db = Database()
    db.connect()
    print db
