#!/bin/bash

# Prevent direct execution:
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

# Check if running on NERSC perlmutter:
# ===========================================================
if [[ "$NERSC_HOST" == "perlmutter" ]]; then
    echo "Running on NERSC Perlmutter"
    MPI_CMD="srun"
else
    echo "Running from" $HOSTNAME
    MPI_CMD="mpirun"
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
if [[ "$DEBUG_CQL3D" == "1" ]]; then
    echo "DEBUG_CQL3D is set to 1. Running CQL3D in debug mode with ddt..."
    cd $CQL3D_RUN_DIR
    time ddt --connect $MPI_CMD -n $NUM_PROCS $CQL3DM_DIR/$EXECUTABLE
    cd $START_DIR
else
    cd $CQL3D_RUN_DIR
    if [[ "$CLI_TO_TXT" == "1" ]]; then
        echo "CLI_TO_TXT is set to 1. Recording CLI output to text file log.txt ..."
        time $MPI_CMD -n $NUM_PROCS $CQL3DM_DIR/xcql3dm_mpi.gfortran64 > log.txt 2>&1 &
    elif [[ "$CLI_TO_TXT" == "0" ]]; then
        echo "CLI_TO_TXT is set to 0. NOT Recording CLI output ..."
        time $MPI_CMD -n $NUM_PROCS $CQL3DM_DIR/xcql3dm_mpi.gfortran64
    elif [[ -z "$CLI_TO_TXT" ]]; then
        echo "CLI_TO_TXT is empty. Do not record CLI output to text file ..."
        time $MPI_CMD -n $NUM_PROCS $CQL3DM_DIR/xcql3dm_mpi.gfortran64
    else
        echo "CLI_TO_TXT has an unexpected value: $CLI_TO_TXT"
        echo "CQL3D run has been terminated ..."
        # Handle any other unexpected values if necessary
    fi
    cd $START_DIR
fi
