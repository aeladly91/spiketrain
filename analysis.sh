#!/usr/bin/env bash

FILENAME="data/spikeTimes_medium"
BIN_SIZE=".5"
MATLAB_PATH="/Applications/MATLAB_R2018b.app/bin/matlab"

python bin.py $FILENAME $BIN_SIZE # creates matlab data with correct binning
$MATLAB_PATH -nodesktop binning
