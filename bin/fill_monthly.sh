#!/usr/bin/env bash

umask 002

date
echo "Preparing to prune and fill archive ..."

SRCDIR=/data/des51.b/data/DTS/src/
cd $SRCDIR

# Setup the cvmfs script
source /cvmfs/des.opensciencegrid.org/users/kadrlica/gridsetup.sh

# Really, just need numpy and psycopg2...
setup readlinePython 6.2.4.1+8
setup astropy 0.4.2+2
setup requests 2.7.0+1
setup finalcut Y2A1+2

# Standalone archive code
VERSION=master
export DECAM_ARCHIVE=$SRCDIR/decam_archive/$VERSION
export PYTHONPATH=$DECAM_ARCHIVE:$PYTHONPATH
export PATH=$DECAM_ARCHIVE/bin:$PATH
export PATH=/home/s1/kadrlica/bin:$PATH # for csub

prune_archive --outdir=$SRCDIR
fill_archive --outdir=$SRCDIR

exit