#!/usr/bin/env bash

if [ "$1" == "-h" ] ; then echo "Usage: fill_nightly.sh <YYYYMMDD>" ; exit 1 ; fi
if [ "$1" == "-help" ] ; then echo "Usage: fill_nightly.sh <YYYYMMDD>" ; exit 1 ; fi
if [ "$1" == "--help" ] ; then echo "Usage: fill_nightly.sh <YYYYMMDD>" ; exit 1 ; fi

STARTDATE=`date --date="1 day ago"  +%Y%m%d`
if [ $# == 1 ] ; then  STARTDATE=$1 ; fi

umask 002

echo "Getting data for night $STARTDATE ..."

SRCDIR=/data/des51.b/data/DTS/src/
cd $SRCDIR

# First grab data from NCSA...
wget -X . -r -A DECam_\*fits.fz  -np --level=2 --no-check-certificate -N -nH --cut-dirs=4 --progress=dot -e dotbytes=4M https://desar2.cosmology.illinois.edu/DESFiles/desarchive/DTS/raw/${STARTDATE}/

# Setup conda
export PATH=/cvmfs/des.opensciencegrid.org/fnal/anaconda2/envs/default/bin:$PATH

# Standalone archive code
VERSION=master
export DECAM_ARCHIVE=$SRCDIR/decam_archive/$VERSION
export PYTHONPATH=$DECAM_ARCHIVE:$PYTHONPATH
export PATH=$DECAM_ARCHIVE/bin:$PATH
export PATH=/home/s1/kadrlica/bin:$PATH # for csub

# Try moving back an additional day
STARTDATE=`date --date="2 days ago"  +%Y%m%d`
fill_night --date=$STARTDATE --outdir=$SRCDIR -v

exit