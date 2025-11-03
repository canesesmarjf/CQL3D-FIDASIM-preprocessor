#!/bin/bash

# RUN directory and id:
export RUN_DIR=$PWD
export RUN_ID=$(basename "$RUN_DIR")

# Computational resources requested:
export FIDASIM_NUM_THREADS
export CQL3D_NUM_PROCS

# Preprocessor options:
export PREPROCESSOR_PLOT_CREATE
export PREPROCESSOR_PLOT_SAVE
export PREPROCESSOR_PLOT_SHOW

# Location of repos:
export FIDASIM_DIR
export CQL3DM_DIR
export PREPROCESSOR_DIR

# Executables to use:
export FIDASIM_EXECUTABLE
export CQL3D_EXECUTABLE

# Debug vars (system dependent):
export FIDASIM_DEBUG
export CQL3D_DEBUG
if [[ -z "$NERSC_HOST" ]]; then
    export FORGE_DIR
    export PATH=$FORGE_DIR:$PATH
fi


