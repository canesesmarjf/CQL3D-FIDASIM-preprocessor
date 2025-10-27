#!/bin/bash

# USER INPUTS:
# ============================================================
# ======================= USER INPUTS ========================
# ============================================================

# Computational resources:
FIDASIM_NUM_THREADS=16

# Select FIDASIM processes to run in the workflow:
FIDASIM_RUN_PREPROCESSOR=1
FIDASIM_RUN_EXEC=1
FIDASIM_RUN_SRCS_TO_TXT=1

# Output options:
PREPROCESSOR_PLOT_CREATE=1 # Create figures
PREPROCESSOR_PLOT_SHOW=1 # Show figures on screen
PREPROCESSOR_PLOT_SAVE=1 # Save figures

# Environment:
FIDASIM_DIR="/home/jfcm/Repos/FIDASIM"
PREPROCESSOR_DIR="/home/jfcm/Repos/CQL3D-FIDASIM-preprocessor"

# Executable:
FIDASIM_EXECUTABLE=fidasim

# Debug options:
FIDASIM_DEBUG=0
FORGE_DIR="/home/jfcm/linaro/forge/24.0.2/bin"

# ============================================================
# ======================= USER INPUTS ========================
# ============================================================

# Run FIDASIM workflow:
python_package="cql3d_to_fidasim_preprocessor/"
source $PREPROCESSOR_DIR/$python_package"activate_conda_env.sh"
source $PREPROCESSOR_DIR/$python_package"export_variables.sh"
source $PREPROCESSOR_DIR/$python_package"Launch_FIDASIM_workflow.sh"
