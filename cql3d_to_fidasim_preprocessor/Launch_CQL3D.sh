#!/bin/bash

# Prevent direct execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This shell script (Launch_CQL3D.sh) cannot be run directly. It must be sourced or executed by another shell script which provides the required env vars."
    echo "Environmental variables required are the following:"
    echo ""
    echo "RUN_ID"
    echo "NUM_THREADS"
    echo "NUM_PROCS"
    echo ""
    echo "To run this script, Use step_1_run_cql3dm.sh or a similar script"
    exit 1
fi

# Export environmental variables:
# ===========================================================
# We need to make these env variables available to the shell processes spawned by CQL3D
# This is how we pass input data to the FIDASIM scripts spawned from within CQL3D
export RUN_ID
export FIDASIM_RUN_DIR="$PWD/$RUN_ID"
export CQL3D_RUN_DIR="$PWD/$RUN_ID"
export START_DIR=$PWD
export NUM_THREADS
export PREPROCESSOR_PLOT

# Run CQL3D:
# ===========================================================
# Initialize run and time it:
cd $CQL3D_RUN_DIR
if [[ "$CLI_TO_TXT" == "1" ]]; then
    echo "CLI_TO_TXT is set to 1. Recording CLI output to text file log.txt ..."
    time mpirun -n $NUM_PROCS $CQL3DM_DIR/xcql3dm_mpi.gfortran64 > log.txt 2>&1 &
elif [[ "$CLI_TO_TXT" == "0" ]]; then
      echo "CLI_TO_TXT is set to 0. NOT Recording CLI output ..."
    time mpirun -n $NUM_PROCS $CQL3DM_DIR/xcql3dm_mpi.gfortran64
elif [[ -z "$CLI_TO_TXT" ]]; then
    echo "CLI_TO_TXT is empty. Do not record CLI output to text file ..."
    time mpirun -n $NUM_PROCS $CQL3DM_DIR/xcql3dm_mpi.gfortran64
else
    echo "CLI_TO_TXT has an unexpected value: $CLI_TO_TXT"
    echo "CQL3D run has been terminated ..."
    # Handle any other unexpected values if necessary
fi
cd $START_DIR
