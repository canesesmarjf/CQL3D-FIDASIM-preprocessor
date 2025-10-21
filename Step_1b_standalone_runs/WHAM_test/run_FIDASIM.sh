#!/bin/bash

# USER INPUTS:
# ============================================================
# ============================================================
# ======================= USER INPUTS ========================
# ============================================================
# ============================================================

# Number of threads:
NUM_THREADS=16

# FIDASIM processes to run in the workflow:
FIDASIM_RUN_PREPROCESSOR=1
FIDASIM_RUN_EXEC=1
FIDASIM_RUN_SRCS_TO_TXT=1

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
RUN_ID=$PWD
FIDASIM_RUN_DIR=$PWD
CQL3D_RUN_DIR=$PWD

# Activate conda environment:
if [[ "$NERSC_HOST" == "perlmutter" ]]; then
  echo "Running on Perlmutter"
else
  echo "Running from" $HOSTNAME
  source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate FIDASIM_env

# Run FIDASIM workflow:
python_package="cql3d_to_fidasim_preprocessor/"
source $PREPROCESSOR_DIR/$python_package"Launch_FIDASIM_workflow.sh"
