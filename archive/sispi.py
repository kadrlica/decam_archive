#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

import numpy as np
import pandas as pd

from archive.database import Database
from archive.utils import date2nite

class SISPI(Database):
    """ SISPI database access. """

    def __init__(self):
        super(SISPI,self).__init__(dbname='db-fnal')
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

def expnum2nite(expnum):
    """Standalone call to SISPI.expnum2nite.

    Paramaters:
    -----------
    expnum : Exposure number(s) to get nite(s) for.
    
    Returns: 
    --------
    nite   : Nite(s) corresponding to exposure number(s)
    """
    return SISPI().expnum2nite(expnum)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
