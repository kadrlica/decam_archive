#!/usr/bin/env bash

STOPDATE=`date +%Y%m%d`
if [ $# == 1 ] ; then  STOPDATE=$1 ; fi

umask 002

date
echo "Preparing to fill archive..."

SRCDIR=/data/des51.b/data/DTS/src/
OUTDIR=$SRCDIR

cd $OUTDIR

export PATH=/cvmfs/des.opensciencegrid.org/fnal/anaconda2/envs/default/bin:$PATH

# Standalone archive code
VERSION=master
export DECAM_ARCHIVE=$SRCDIR/decam_archive/$VERSION
export PYTHONPATH=$DECAM_ARCHIVE:$PYTHONPATH
export PATH=$DECAM_ARCHIVE/bin:$PATH
export PATH=/home/s1/kadrlica/bin:$PATH # for csub

#echo "Pruning archive..."
#prune_archive --outdir=$SRCDIR

echo "Filling archive..."
# Fill the entire archive
fill_archive -v --njobs 4 --outdir=$OUTDIR --tstop $STOPDATE --exptime 30 --logdir log/download

# Just a single propid
#PROPID=2020B-0282 #2019A-0305 #2020B-0282 #2020A-0399 #2020B-0241
#fill_archive -v --njobs 5 --outdir=$OUTDIR --propid $PROPID

## The `-n 1` argument delays subsequent jobs until fill_archive is done
 
# The DESDM exposures are not always in the right nite
echo "Fixing nite..."
csub -o log/cron_fix_nite.log -n 1 fix_nite -v --outdir=$SRCDIR

# We may have downloaded some bad exposures
#echo "Pruning archive again..."
#csub -o log/cron_prune_archive.log -n 5 prune_archive --outdir=$SRCDIR

# Link back to the main directory
echo "Linking nite to $SRCDIR..."
link_archive --njobs 1 --indir $OUTDIR --outdir $SRCDIR
 
## Load the exposure table
echo "Submitting load_exposure_table..."
csub -o log/cron_load_exposure.log -n 1 load_exposure_table

exit
