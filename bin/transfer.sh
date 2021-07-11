#!/usr/bin/env bash

# 2015, 2016 kept on des91.b
DATE=2015 # 2016
SRCDIR=/data/des51.b/data/DTS/src
OUTDIR=/data/des91.b/data/DTS/src

# 2017, 2018 kept on des41.b
DATE=201807 #2018 # 2017
SRCDIR=/data/des51.b/data/DTS/src
OUTDIR=/data/des41.b/data/DTS/src

# Find the nites
NITES=$(find $SRCDIR/${DATE}* -maxdepth 0  ! -type l)

# Move the data from SRCDIR to OUTDIR
cd $OUTDIR
for nite in $NITES; do 
    csub -v -n 8 mv $nite .; 
done

# Symlink to transferred directories
cd $SRCDIR
csub -n 1 ln -s $OUTDIR/${DATE}\*/ .
cd -
