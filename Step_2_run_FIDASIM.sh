#!/bin/bash

# This script allows one to run FIDASIM using the default input file in one of the following modes:
# - OPENMP
# - OPENMP + DEBUG
# - MPI
# - MPI + DEBUG

# USAGE:
# ./Step_2_run_FIDASIM.sh [OPENMP or MPI] [DEBUG]

# Shell functions:
# ==========================================================================
# Confirm user's selections:
confirm_selections() {
    echo " "
    read -p "Are these selections correct? (y/n): " response
    echo " "
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Proceeding with run"
    else
        echo "Operation aborted by the user."
        exit 1
    fi
}

#Defaults:
# ==========================================================================
# Set default input file:
INPUT_FILE_ROOT=$(pwd)
# INPUT_FILE_NAME=/Step_1_output/WHAM_example_inputs.dat
INPUT_FILE_NAME=/Step_1b_output/ITER_neutral_wall_src_inputs.dat
INPUT_FILE=$INPUT_FILE_ROOT$INPUT_FILE_NAME

# Set default values:
USE_OPENMP="y"
USE_MPI="n"
DEBUG="n"
NUM_PROCS=4
NUM_THREADS=18

# Parse command-line arguments:
# ==========================================================================
for arg in "$@"
do
    case $arg in
        OPENMP)
        USE_OPENMP="y"
        echo "USE_OPENMP=$USE_OPENMP"
        ;;
        MPI)
        USE_MPI="y"
        echo "USE_MPI=$USE_MPI"
        ;;
        DEBUG)
        DEBUG="y"
        echo "DEBUG=$DEBUG"
        ;;
        *)
        echo "Invalid argument: $arg"
        echo "Usage: $0 [OPENMP] [MPI] [DEBUG]"
        exit 1
        ;;
    esac
done

# If no arguments are provided, default to OPENMP=y
if [ "$#" -eq 0 ]; then
    USE_OPENMP="y"
    echo "Using default mode: OPENMP"
    echo " "
    echo "Usage: $0 [OPENMP] [MPI] [DEBUG]"
    echo " "
fi

echo " "
echo "INPUT_FILE_ROOT=$INPUT_FILE_ROOT"
echo "INPUT_FILE_NAME=$INPUT_FILE_NAME"
echo " "

# Define the directory where the Forge binaries are located
FORGE_DIR=/home/jfcm/linaro/forge/24.0.2/bin

# Export FORGE_DIR to the PATH
export PATH=$FORGE_DIR:$PATH

# Path to the FIDASIM executable
EXECUTABLE=/home/jfcm/Repos/FIDASIM/fidasim

# Run FIDASUM based on settings:
# ==========================================================================
if [ "$USE_MPI" == "y" ]; then
    echo "Running with MPI with NUM_PROCS=$NUM_PROCS"

    if [ "$DEBUG" == "y" ]; then
        echo "Running in DEBUG mode (reverse connect to FORGE)..."

        confirm_selections
        ddt --connect mpirun -n $NUM_PROCS $EXECUTABLE $INPUT_FILE
    else
        confirm_selections
        mpirun -n $NUM_PROCS $EXECUTABLE $INPUT_FILE
    fi

elif [ "$USE_OPENMP" == "y" ]; then
    echo "Running with OpenMP with NUM_THREADS=$NUM_THREADS"

    if [ "$DEBUG" == "y" ]; then
        echo "Running in DEBUG mode (reverse connect to FORGE)..."

        confirm_selections
        ddt --connect $EXECUTABLE $INPUT_FILE $NUM_THREADS
    else
        confirm_selections
        $EXECUTABLE $INPUT_FILE $NUM_THREADS
    fi
fi

# After running FIDASIM, automatically run Step_3_save_birth_txt.py
echo " "
echo "FIDASIM run completed. Running Step_3_save_birth_txt.py..."
python Step_3_save_birth_txt.py
echo " "
echo "Post-process script completed."
