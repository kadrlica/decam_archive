#!/usr/bin/env python
"""
Argument parsing and logging.
"""
__author__ = "Alex Drlica-Wagner"

import logging
import argparse
import dateutil

from archive import __version__

class SpecialFormatter(logging.Formatter):
    """
    Class for overloading log formatting based on level.
    """
    FORMATS = {'DEFAULT'       : "%(message)s",
               logging.WARNING : "WARNING: %(message)s",
               logging.ERROR   : "ERROR: %(message)s",
               logging.DEBUG   : "DEBUG: %(message)s"}
 
    def format(self, record):
        self._fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        return logging.Formatter.format(self, record)

def setup_logging(level=logging.INFO):
    # Can we update the handler rather than adding one?
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(SpecialFormatter())
    if not len(logger.handlers):
        logger.addHandler(handler)
    logger.setLevel(level)

setup_logging()

class VerboseAction(argparse._StoreTrueAction):
    """
    Class for setting logging level from verbosity.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        super(VerboseAction,self).__call__(parser, namespace, values, option_string)
        if self.const: logging.getLogger().setLevel(logging.DEBUG)

class DatetimeAction(argparse.Action):
    """
    Class for setting logging level from verbosity.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        datetime = dateutil.parser.parse(values)
        setattr(namespace, self.dest, datetime)


class Parser(argparse.ArgumentParser):
    def __init__(self,*args,**kwargs):
        super(Parser,self).__init__(*args,**kwargs)
        self.add_argument('-v','--verbose',action=VerboseAction,
                          help='output verbosity')
        self.add_argument('--version', action='version',
                          version='maglites v'+__version__,
                          help="print version number and exit")
        self.add_argument('--dryrun',action='store_true',
                          help='dry run (do nothing)')


class LoaderParser(Parser):
    def __init__(self,*args,**kwargs):
        super(LoaderParser,self).__init__(*args,**kwargs)
        self.add_argument('expnum',const=None,nargs='*',type=int,
                          help='explicit exposures to load')
        self.add_argument('--index',action='store_true',
                          help='create table indexes')
        self.add_argument('-f','--force',action='store_true',
                          help='overwrite existing database entries')
        self.add_argument('-k','--chunk',type=int,default=100,
                          help='chunk size for upload')
        self.add_argument('-n','--nproc',type=int,default=20,
                          help='number of parallel processes')


if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = Parser()
    args = parser.parse_args()
