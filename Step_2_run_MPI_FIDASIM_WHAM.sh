#!/bin/bash

# Number of processors:
NUM_PROCS=3

# Path to the FIDASIM executable and input file
EXECUTABLE=/home/jfcm/Repos/FIDASIM/fidasim
INPUT_FILE=./Step_1_output/WHAM_example_inputs.dat

# Run FIDASIM:
mpirun -n $NUM_PROCS $EXECUTABLE $INPUT_FILE
