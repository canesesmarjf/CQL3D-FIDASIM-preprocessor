#!/bin/bash

# USER INPUTS:
# ============================================================
# ============================================================
# ======================= USER INPUTS ========================
# ============================================================
# ============================================================

# Define run ID:
#RUN_ID="WHAM_high_ne_nonthermal_multistep_cx"
RUN_ID="WHAM_NBI_kunal"

# Number of threads:
NUM_THREADS=20

# FIDASIM processes to run in the workflow:
FIDASIM_RUN_PREPROCESSOR=1
FIDASIM_RUN_EXEC=1
FIDASIM_RUN_SRCS_TO_TEXT=0

# If running preprocessor, enable plotting output:
PREPROCESSOR_PLOT=1

# Enable debugging FIDASIM with linaro forge ddt:
FIDASIM_DEBUG=0

# ============================================================
# ============================================================
# ======================= USER INPUTS ========================
# ============================================================
# ============================================================

# Define run directories relative to the parent directory using $PWD.
FIDASIM_RUN_DIR=$PWD/$RUN_ID
CQL3D_RUN_DIR=$PWD/$RUN_ID

# Run FIDASIM workflow:
python_package="cql3d_to_fidasim_preprocessor/"
source $PREPROCESSOR_DIR/$python_package"Launch_FIDASIM_workflow.sh"
