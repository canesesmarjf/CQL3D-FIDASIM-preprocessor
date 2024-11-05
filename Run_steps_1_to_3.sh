#!/bin/bash

# USER INPUTS:
# ============================================================
# ======================= USER INPUTS ========================
# ============================================================

# Case to run [required]:
#RUN_ID="WHAM_example_2"
#RUN_ID="WHAM_no_f4d"
#FIDASIM_RUN_DIR=$PWD/fidasim_files/$RUN_ID
#CQL3D_RUN_DIR=$PWD/cql3d_files/$RUN_ID

# Diagnostics:
echo " FIDASIM_RUN_DIR: $FIDASIM_RUN_DIR"
echo " CQL3D_RUN_DIR: $CQL3D_RUN_DIR"

# Repo location [optional]:
fidasim_DIR=""
#pp_DIR="../../"

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
    echo " pp_DIR: $pp_DIR"
fi

# Step 1: Run preprocessor script to produce input files for FIDASIM:
# ===================================================================
# Requires the following:
# 1- [file] run_id_config.nml (FIDASIM run configuration file)
# 2- [file] run_id_cql_config.nml (CQL3D run configuration file)
# 3- [repo] cql3d-fidasim-preprocessor (python)
# 4- [repo] FIDASIM installation (fortran)
# 5- [file] cqlinput
# 6- [file] eqdsk

# If running in coupled manner, CQL3D will produce the following files needed for this step:
# 7- mnemonic.nc
# 8- mnemonic_f4d_001.nc
# 9- mnemonic_f4d_001.nc

# The output product is the follownig:
# - run_id_distribution.h5
# - run_id_equilibrium.h5
# - run_id_geometry.h5
# - run_id_inputs.dat

STEP_1="python $pp_DIR/Step_1_assemble_FIDASIM_inputs.py"
$STEP_1 --fida-run-dir $FIDASIM_RUN_DIR --cql-run-dir $CQL3D_RUN_DIR --fidasim-dir $fidasim_DIR --plot
if [ $? -ne 0 ]; then
    echo "Error: Step_1_assemble_FIDASIM_inputs.py failed"
    exit 1
fi

# Step 2: Run FIDASIM:
# ============================================================
STEP_2="$pp_DIR/Step_2_run_FIDASIM.sh"
$STEP_2 --fida-run-dir $FIDASIM_RUN_DIR --parallel-mode openmp -n $NUM_THREADS --executable $fidasim_DIR/fidasim --verbose
if [ $? -ne 0 ]; then
    echo "Error: Step_2_run_FIDASIM.sh failed"
    exit 1
fi

# Step 3: Save sources and sinks into text file (.dat):
# ============================================================
STEP_3="python $pp_DIR/Step_3_save_sources_and_sinks.py"
$STEP_3 --fida-run-dir $FIDASIM_RUN_DIR --cql-run-dir $CQL3D_RUN_DIR
if [ $? -ne 0 ]; then
    echo "Error: Step_3_save_sources_and_sinks.py failed"
    exit 1
fi

echo "All FIDASIM steps completed successfully!"
sleep 6
