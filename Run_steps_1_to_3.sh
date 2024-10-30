#!/bin/bash

# USER INPUTS:
# ============================================================
# ======================= USER INPUTS ========================
# ============================================================

# Case to run [required]:
#RUN_ID="WHAM_example_2"
RUN_ID="WHAM_no_f4d"
FIDASIM_RUN_DIR=./fidasim_files/$RUN_ID
CQL3D_RUN_DIR=./cql3d_files/$RUN_ID

# Repo location [optional]:
fidasim_DIR=""
pp_DIR="./"

# ============================================================
# ======================= USER INPUTS ========================
# ============================================================


# If repo locations are not set, use system variables:
# ============================================================
# This will work only if repo env vars have been set in .bashrc file and sourced
if [ -z "$fidasim_DIR" ]; then
    fidasim_DIR=$FIDASIM_DIR
fi
if [ -z "$pp_DIR" ]; then
    pp_DIR=$PREPROCESSOR_DIR
fi

# Step 1: Run preprocessor script to produce input files for FIDASIM:
# ============================================================
python $pp_DIR/Step_1_assemble_FIDASIM_inputs.py --run-dir $FIDASIM_RUN_DIR --fidasim-dir $fidasim_DIR --plot
if [ $? -ne 0 ]; then
    echo "Error: Step_1_assemble_FIDASIM_inputs.py failed"
    exit 1
fi

# Step 2: Run FIDASIM:
# ============================================================
$pp_DIR/Step_2_run_FIDASIM.sh --run-dir $FIDASIM_RUN_DIR --parallel-mode openmp -n 14 --executable $fidasim_DIR/fidasim --verbose
if [ $? -ne 0 ]; then
    echo "Error: Step_2_run_FIDASIM.sh failed"
    exit 1
fi

# Step 3: Save sources and sinks into text file (.dat):
# ============================================================
python $pp_DIR/Step_3_save_sources_and_sinks.py --run-dir $FIDASIM_RUN_DIR
if [ $? -ne 0 ]; then
    echo "Error: Step_3_save_sources_and_sinks.py failed"
    exit 1
fi

echo "All FIDASIM steps completed successfully!"