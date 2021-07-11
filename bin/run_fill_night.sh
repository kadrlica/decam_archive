#!/usr/bin/env bash

if [ "$1" == "-h" ]     ; then echo "Usage: run_fill_night.sh <YYYYMMDD>" ; exit 1 ; fi
if [ "$1" == "-help" ]  ; then echo "Usage: run_fill_night.sh <YYYYMMDD>" ; exit 1 ; fi
if [ "$1" == "--help" ] ; then echo "Usage: run_fill_night.sh <YYYYMMDD>" ; exit 1 ; fi

STARTDATE=`date --date="1 day ago"  +%Y%m%d`
if [ $# == 1 ] ; then  STARTDATE=$1 ; fi

umask 002

echo "Getting data for night ${STARTDATE}..."

SRCDIR=/data/des51.b/data/DTS/src/
OUTDIR=$SRCDIR

cd $OUTDIR

# First grab data from NCSA...
#url=https://desar2.cosmology.illinois.edu/DESFiles/desarchive/DTS/raw
#wget -X . -r -A DECam_\*fits.fz  -np --level=2 --no-check-certificate -N -nH --cut-dirs=4 --progress=dot -e dotbytes=4M $url/${STARTDATE}/

url=http://decade.ncsa.illinois.edu/deca_archive/RAW
#wget -X . -r -A DECam_\*fits.fz  -np --level=2 --no-check-certificate -N -nH --cut-dirs=2 --progress=dot -e dotbytes=4M $url/${STARTDATE}/

# Setup conda
export PATH=/cvmfs/des.opensciencegrid.org/fnal/anaconda2/envs/default/bin:$PATH

# Standalone archive code
VERSION=master
export DECAM_ARCHIVE=$SRCDIR/decam_archive/$VERSION
export PYTHONPATH=$DECAM_ARCHIVE:$PYTHONPATH
export PATH=$DECAM_ARCHIVE/bin:$PATH
export PATH=/home/s1/kadrlica/bin:$PATH # for csub

# Default with no propid specified
fill_night --njobs 5 --date=$STARTDATE --outdir=$OUTDIR -v 

# To target one or more propids:
#fill_night --njobs 5 --outdir=$OUTDIR -v --propid="2020B-0282"
    

echo "Linking nite to $SRCDIR..."
link_archive --njobs 5 --indir $OUTDIR --outdir $SRCDIR

# Load the exposure table
echo "Submitting load_exposure_table..."
csub -o log/cron_load_exposure.log -n 5 load_exposure_table

exit
