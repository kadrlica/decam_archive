#!/usr/bin/env bash

if [ "$1" == "-h" ] ; then echo "Usage: fill_quickly.sh <YYYYMMDD>" ; exit 1 ; fi
if [ "$1" == "-help" ] ; then echo "Usage: fill_quickly.sh <YYYYMMDD>" ; exit 1 ; fi
if [ "$1" == "--help" ] ; then echo "Usage: fill_quickly.sh <YYYYMMDD>" ; exit 1 ; fi

STARTDATE=`date --date="12 hours ago"  +%Y%m%d`
if [ $# == 1 ] ; then  STARTDATE=$1 ; fi

umask 002

echo "Getting data for night ${STARTDATE}..."

SRCDIR=/data/des51.b/data/DTS/src
OUTDIR=$SRCDIR

cd $OUTDIR

# Grab data from NCSA DESDM...
#URL=https://desar2.cosmology.illinois.edu/DESFiles/desarchive/DTS/raw
#wget -X . -r -A DECam_\*fits.fz  -np --level=2 --no-check-certificate -N -nH --cut-dirs=4 --progress=dot -e dotbytes=4M ${URL}/${STARTDATE}/
#wget -X . -r -A DECam_008804\*fits.fz  -np --level=2 --no-check-certificate -N -nH --cut-dirs=4 --progress=dot -e dotbytes=4M ${URL}/${STARTDATE}/

# Grab data from NCSA DECADE...
URL=http://decade.ncsa.illinois.edu/deca_archive/RAW
wget -X . -r -A DECam_\*fits.fz  -np --level=2 --no-check-certificate -N -nH --cut-dirs=2 --progress=dot -e dotbytes=4M ${URL}/${STARTDATE}/

# Setup conda
export PATH=/cvmfs/des.opensciencegrid.org/fnal/anaconda2/envs/default/bin:$PATH

# Standalone archive code
VERSION=master
export DECAM_ARCHIVE=$SRCDIR/decam_archive/$VERSION
export PYTHONPATH=$DECAM_ARCHIVE:$PYTHONPATH
export PATH=$DECAM_ARCHIVE/bin:$PATH
export PATH=/home/s1/kadrlica/bin:$PATH # for csub

# Grab from NOAO staging using lftp
retrieve-too-data.sh -n ${STARTDATE}

# Don't move back an additional day
#STARTDATE=`date --date="$STARTDATE - 1 day"  +%Y%m%d`
fill_night --njobs 5 --date=$STARTDATE --outdir=$OUTDIR -v

#fill_night --njobs 5 --date=$STARTDATE --outdir=$OUTDIR -v --propid="2019A-0235"
#fill_night --njobs 5 --date=$STARTDATE --outdir=$OUTDIR -v \
#    --propid="2018A-0242" --propid="2019A-0272" \
#    --cert $SRCDIR/decam_archive/certificates/drlicawagnera-20190728.cert

#echo "Linking nite to $SRCDIR..."
#link_archive --njobs 5 --indir $OUTDIR --outdir $SRCDIR

# Don't load the exposure table
#echo "Submitting load_exposure_table..."
#csub -o log/cron_load_exposure.log -n 5 load_exposure_table

exit