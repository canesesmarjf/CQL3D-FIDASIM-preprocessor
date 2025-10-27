#!/bin/bash

# Prevent direct execution:
# =========================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script (${BASH_SOURCE[0]##*/}) must be *sourced*, not executed."
    exit 1
fi

# Select MPI run command:
# ===========================================================
if [[ "$NERSC_HOST" == "perlmutter" ]]; then
    echo "Running on NERSC Perlmutter"
    MPI_CMD="srun"

else
    echo "Running from" $HOSTNAME
    MPI_CMD="mpirun"
fi

# Run CQL3D:
# ===========================================================
RUN_CMD="$MPI_CMD -n $CQL3D_NUM_PROCS $CQL3DM_DIR/$CQL3D_EXECUTABLE"

if [[ "$CQL3D_DEBUG" == "1" ]]; then
    echo "CQL3D_DEBUG is set to 1. Running CQL3D in debug mode with ddt..."
    time ddt --connect $RUN_CMD
else
    if [[ "$CLI_TO_TXT" == "1" ]]; then
        echo "CLI_TO_TXT is set to 1. Recording CLI output to text file log.txt ..."
        time $RUN_CMD > log.txt 2>&1 &
    elif [[ "$CLI_TO_TXT" == "0" ]]; then
        echo "CLI_TO_TXT is set to 0. NOT Recording CLI output ..."
        time $RUN_CMD
    else
        echo "CLI_TO_TXT has an unexpected value: $CLI_TO_TXT"
        echo "CQL3D run has been terminated ..."
    fi
fi
