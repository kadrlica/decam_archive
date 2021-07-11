#!/usr/bin/env bash

if [ "$1" == "-h" ] ; then echo "Usage: run_fill_quick.sh <PROPID>" ; exit 1 ; fi
if [ "$1" == "-help" ] ; then echo "Usage: run_fill_quick.sh <PROPID>" ; exit 1 ; fi
if [ "$1" == "--help" ] ; then echo "Usage: run_fill_quick.sh <PROPID>" ; exit 1 ; fi

# Default PROPIDs
PROPIDS="2020B-0053 2021A-0275 2021A-0246 2021A-0149 2019A-0305"
if [ $# == 1 ] ; then  PROPIDS=$1 ; fi

umask 002

date
echo "Getting data for propid: ${PROPIDS}..."

SRCDIR=/data/des51.b/data/DTS/src
OUTDIR=$SRCDIR

cd $OUTDIR

# Setup conda
export PATH=/cvmfs/des.opensciencegrid.org/fnal/anaconda2/envs/default/bin:$PATH

# Standalone archive code
VERSION=master
export DECAM_ARCHIVE=$SRCDIR/decam_archive/$VERSION
export PYTHONPATH=$DECAM_ARCHIVE:$PYTHONPATH
export PATH=$DECAM_ARCHIVE/bin:$PATH
export PATH=/home/s1/kadrlica/bin:$PATH # for csub

for PROPID in ${PROPIDS}; do
    echo "Downloading propid: ${PROPID}"
    fill_archive -v -a noir --njobs 2 --outdir=$OUTDIR  --propid=$PROPID --exptime 5 --table ./table/archive_${PROPID}.npy --logdir log/download
done

# Don't load the exposure table now...
#echo "Submitting load_exposure_table..."
#csub -o log/cron_load_exposure.log -n 5 load_exposure_table

exit
