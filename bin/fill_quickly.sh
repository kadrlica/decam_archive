#!/usr/bin/env bash

if [ "$1" == "-h" ] ; then echo "Usage: fill_nightly.sh <YYYYMMDD>" ; exit 1 ; fi
if [ "$1" == "-help" ] ; then echo "Usage: fill_nightly.sh <YYYYMMDD>" ; exit 1 ; fi
if [ "$1" == "--help" ] ; then echo "Usage: fill_nightly.sh <YYYYMMDD>" ; exit 1 ; fi

STARTDATE=`date --date="12 hours ago"  +%Y%m%d`
if [ $# == 1 ] ; then  STARTDATE=$1 ; fi

umask 002

echo "Getting data for night ${STARTDATE}..."

SRCDIR=/data/des51.b/data/DTS/src/
cd $SRCDIR

# Grab data from NCSA...
wget -X . -r -A DECam_\*fits.fz  -np --level=2 --no-check-certificate -N -nH --cut-dirs=4 --progress=dot -e dotbytes=4M https://desar2.cosmology.illinois.edu/DESFiles/desarchive/DTS/raw/${STARTDATE}/
