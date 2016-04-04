#!/usr/bin/env bash

if [ "$1" == "-h" ] ; then echo "Usage: get_data <YYYYMMDD>" ; exit 1 ; fi
if [ "$1" == "-help" ] ; then echo "Usage: get_data <YYYYMMDD>" ; exit 1 ; fi
if [ "$1" == "--help" ] ; then echo "Usage: get_data <YYYYMMDD>" ; exit 1 ; fi

STARTDATE=`date --date="1 day ago"  +%Y%m%d`
if [ $# == 1 ] ; then  STARTDATE=$1 ; fi

umask 002

echo "Getting data for night $STARTDATE"

SRCDIR=/data/des51.b/data/DTS/src/
cd $SRCDIR

# First grab data from NCSA...
wget  -X . -r -A DECam_\*fits.fz  -np --level=2 --no-check-certificate -N -nH --cut-dirs=4 https://desar2.cosmology.illinois.edu/DESFiles/desarchive/DTS/raw/${STARTDATE}/

# Setup the cvmfs script
source /cvmfs/des.opensciencegrid.org/users/kadrlica/gridsetup.sh

# Really, just need numpy and psycopg2...
setup readlinePython 6.2.4.1+8
setup astropy 0.4.2+2
setup requests 2.7.0+1
setup finalcut Y2A1+2

# Standalone archive code
VERSION=v0.1.1
export DECAM_ARCHIVE=$SRCDIR/decam_archive/$VERSION
export PYTHONPATH=$DECAM_ARCHIVE:$PYTHONPATH
export PATH=$DECAM_ARCHIVE/bin:$PATH

fill_night --date=$STARTDATE --outdir=$SRCDIR -v

exit