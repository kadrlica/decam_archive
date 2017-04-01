#!/usr/bin/env bash

umask 002

date
echo "Preparing to prune and fill archive..."

SRCDIR=/data/des51.b/data/DTS/src/
cd $SRCDIR

export PATH=/cvmfs/des.opensciencegrid.org/fnal/anaconda2/envs/default/bin:$PATH

# Standalone archive code
VERSION=master
export DECAM_ARCHIVE=$SRCDIR/decam_archive/$VERSION
export PYTHONPATH=$DECAM_ARCHIVE:$PYTHONPATH
export PATH=$DECAM_ARCHIVE/bin:$PATH
export PATH=/home/s1/kadrlica/bin:$PATH # for csub

prune_archive --outdir=$SRCDIR
fill_archive --outdir=$SRCDIR
prune_archive --outdir=$SRCDIR

# Load the exposure table
echo "Preparing to load exposure table..."
load_exposure_table

exit