#!/bin/bash

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ================= START USER INPUTS: ======================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Define run directory:
# RUN_ID="WHAM_no_f4d"
# RUN_ID="WHAM_test"
RUN_ID='wham_source_test'

# Number of processors:
NUM_PROCS=15 # Number of MPI processes for CQL3D
NUM_THREADS=15 # Number of threads for FIDASIM

# Record terminal output to text file:
CLI_TO_TXT=0

# Enable plotting output from preprocessor:
PREPROCESSOR_PLOT=1

# Debugging parameters:
DEBUG_CQL3D=1
EXECUTABLE=xcql3dm_mpi.gfortran64_DEBUG
FORGE_DIR=/home/jfcm/linaro/forge/24.0.2/bin

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==================== END USER INPUTS: =====================
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Launch CQL3D:
python_package="cql3d_to_fidasim_preprocessor/"
source $PREPROCESSOR_DIR/$python_package"Launch_CQL3D.sh"
