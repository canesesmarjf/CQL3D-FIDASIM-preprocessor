#!/bin/bash

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ================= START USER INPUTS: ======================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Computational resources:
CQL3D_NUM_PROCS=8
FIDASIM_NUM_THREADS=20

# Output options:
CLI_TO_TXT=0 # Record terminal output to text file
PREPROCESSOR_PLOT_CREATE=1 # Create figures
PREPROCESSOR_PLOT_SHOW=1 # Show figures on screen
PREPROCESSOR_PLOT_SAVE=1 # Save figures

# Environment:
#CQL3DM_DIR='/home/jfcm/Repos/CQL3DM/2024_02_27/src'
CQL3DM_DIR='/home/jfcm/Repos/CQL3DM/2025_09_20/src'
FIDASIM_DIR="/home/jfcm/Repos/FIDASIM"
PREPROCESSOR_DIR="/home/jfcm/Repos/CQL3D-FIDASIM-preprocessor"

# Executable:
CQL3D_EXECUTABLE=xcql3dm_mpi.gfortran64
#CQL3D_EXECUTABLE=xcql3dm_mpi.gfortran64_DEBUG
#CQL3D_EXECUTABLE=xcql3d_mpi.perl
FIDASIM_EXECUTABLE=fidasim

# Debug options:
CQL3D_DEBUG=0
FORGE_DIR="/home/jfcm/linaro/forge/24.0.2/bin"

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==================== END USER INPUTS: =====================
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Launch CQL3D:
python_package="cql3d_to_fidasim_preprocessor/"
source $PREPROCESSOR_DIR/$python_package"activate_conda_env.sh"
source $PREPROCESSOR_DIR/$python_package"export_variables.sh"
source $PREPROCESSOR_DIR/$python_package"Launch_CQL3D.sh"
