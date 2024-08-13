#!/bin/bash

# Number of threads:
NUM_THREADS=18

# Path to the FIDASIM executable and input file
EXECUTABLE=/home/jfcm/Repos/FIDASIM/fidasim
INPUT_FILE=./Step_1_output/WHAM_example_inputs.dat

# Run FIDASIM:
$EXECUTABLE $INPUT_FILE $NUM_THREADS
