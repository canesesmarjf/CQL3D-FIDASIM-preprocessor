#!/bin/bash

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ================= START USER INPUTS: ======================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Define run directory:
# RUN_ID="WHAM_no_f4d"
#RUN_ID="WHAM_test"
RUN_ID="WHAM_100keV"

# Number of processors:
NUM_PROCS=13 # Number of MPI processes for CQL3D
NUM_THREADS=8 # Number of threads for FIDASIM

# Record terminal output to text file:
CLI_TO_TXT=0

# Enable plotting output from preprocessor:
PREPROCESSOR_PLOT=1

# CQL3D executable:
#EXECUTABLE=xcql3d_mpi.perl
EXECUTABLE=xcql3dm_mpi.gfortran64

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==================== END USER INPUTS: =====================
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Launch CQL3D:
python_package="cql3d_to_fidasim_preprocessor/"
source $PREPROCESSOR_DIR/$python_package"Launch_CQL3D.sh"
