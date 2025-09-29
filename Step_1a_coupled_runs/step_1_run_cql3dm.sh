#!/bin/bash

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ================= START USER INPUTS: ======================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Define run directory:
RUN_ID="WHAM_test"
#RUN_ID="WHAM_Bob_IAEA_wall"

# Number of processors:
NUM_PROCS=8 # Number of MPI processes for CQL3D
NUM_THREADS=20 # Number of threads for FIDASIM

# Record terminal output to text file:
CLI_TO_TXT=0

# Enable plotting output from preprocessor:
PREPROCESSOR_PLOT=1

# CQL3D executable:
#EXECUTABLE=xcql3d_mpi.perl
#export CQL3DM_DIR='/home/jfcm/Repos/CQL3DM/2024_02_27/src'
export CQL3DM_DIR='/home/jfcm/Repos/CQL3DM/2025_09_20/src'
EXECUTABLE=xcql3dm_mpi.gfortran64

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==================== END USER INPUTS: =====================
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Launch CQL3D:
python_package="cql3d_to_fidasim_preprocessor/"
source $PREPROCESSOR_DIR/$python_package"Launch_CQL3D.sh"
