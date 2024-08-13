#!/bin/bash

# Define the directory where the Forge binaries are located
FORGE_DIR=/home/jfcm/arm/forge/22.0.1/bin

# Export FORGE_DIR to the PATH
export PATH=$FORGE_DIR:$PATH

# Number of MPI processes
NUM_PROCS=2

# Path to the FIDASIM executable and input file
EXECUTABLE=/home/jfcm/Repos/FIDASIM/fidasim
INPUT_FILE_ROOT=/home/jfcm/Documents/compX/RFSCIDAC/FIDASIM_work/Step_4_compute_FIDASIM_input_files
INPUT_FILE_NAME=/Step_1_output_debug/WHAM_example_debug_inputs.dat

# Assemble input file full path:
INPUT_FILE=$INPUT_FILE_ROOT$INPUT_FILE_NAME
# Run DDT in reverse connection mode
ddt --connect mpirun -n $NUM_PROCS $EXECUTABLE $INPUT_FILE
