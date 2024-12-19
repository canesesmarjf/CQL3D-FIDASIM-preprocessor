#!/bin/bash

# USER INPUTS:
# ============================================================
# ============================================================
# ======================= USER INPUTS ========================
# ============================================================
# ============================================================

# Define run ID:
RUN_ID="WHAM_high_ne_nonthermal_multistep_cx"
# RUN_ID="WHAM_high_ne_thermal_multistep_cx"
# RUN_ID="WHAM_high_ne_nonthermal"
# RUN_ID="WHAM_wall_flux_cold_plasma_multistep_cx"
# RUN_ID="WHAM_low_ne_nonthermal"
# RUN_ID="WHAM_low_ne_thermal"
# RUN_ID="WHAM_wall_flux_cold_plasma"
# RUN_ID="WHAM_example_1"

# Number of threads and debugging:
NUM_THREADS=20
DEBUG_FLAG="" #"--debug"

# FIDASIM processes to run in the workflow:
FIDASIM_RUN_PREPROCESSOR=1
FIDASIM_RUN_EXEC=0
FIDASIM_RUN_SRCS_TO_TEXT=0

# If running preprocessor, enable plotting output:
PREPROCESSOR_PLOT=1

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
