#!/bin/bash

echo ""
echo "Starting Launch_FIDASIM_workflow.sh:"
echo "================================================================================================================="
echo ""

# Do not run if env vars have been not been set:
# ==============================================
if [[ -z "$RUN_ID" ]]; then
  echo "ERROR: Env vars not defined ..."
  exit 1
else
  echo "RUN_ID: " $RUN_ID
  echo "RUN_DIR: " $RUN_DIR
fi

# Set default values for preprocessor operations:
# =============================================================
if [[ -z "$FIDASIM_RUN_PREPROCESSOR" ]]; then FIDASIM_RUN_PREPROCESSOR=1; fi
if [[ -z "$FIDASIM_RUN_EXEC" ]]; then FIDASIM_RUN_EXEC=1; fi
if [[ -z "$FIDASIM_RUN_SRCS_TO_TXT" ]]; then FIDASIM_RUN_SRCS_TO_TXT=1; fi
if [[ -z "$FIDASIM_DEBUG" ]]; then FIDASIM_DEBUG=0; fi

# ===================================================================
# STEP 1: Run preprocessor script to produce input files for FIDASIM:
# ===================================================================
# Step 1 requires the following:
# 1- [file] config_fida.nml (FIDASIM run configuration file)
# 3- [repo] cql3d-fidasim-preprocessor (python)
# 4- [repo] FIDASIM installation (fortran)
# 5- [file] cqlinput
# 6- [file] eqdsk
# *- [python env manager] CONDA (needed to setup required python packages

# INPUT:
# The following files from CQL3D are needed for this step, where <mnemonic> refers to the "mnemonic"
# variable in cqlinput file:
# 7- <mnemonic>.nc
# 8- <mnemonic>_f4d_001.nc
# 9- <mnemonic>_f4d_001.nc

# OUTPUT:
# The output product for this step is the following:
# - <RUN_ID>_distribution.h5 (4D distribution function)
# - <RUN_ID>_equilibrium.h5 (equilibrium file: plasma density, neutral gas and magnetic field profiles)
# - <RUN_ID>_geometry.h5 (NBI geometry)
# - <RUN_ID>_inputs.dat (FIDASIM input namelist)

PYTHON_PACKAGE="cql3d_to_fidasim_preprocessor"
if [[ "$FIDASIM_RUN_PREPROCESSOR" == "1" ]]; then
  echo ""
  echo "Step 1: Running A_assemble_FIDASIM_inputs.py"
  echo "--------------------------------------------"

  # Python debugging options:
  DEBUG_CMD="-m pdb "
  DEBUG_CMD=""

  python $DEBUG_CMD $PREPROCESSOR_DIR/$PYTHON_PACKAGE/A_assemble_FIDASIM_inputs.py
  if [ $? -ne 0 ]; then
      echo "Error: A_assemble_FIDASIM_inputs.py failed"
      exit 1
  fi
fi

# ===================================================================
# Step 2: Run FIDASIM:
# ============================================================
# Using the files outputted by STEP 1, we can now run FIDASIM:
if [[ "$FIDASIM_RUN_EXEC" == "1" ]]; then
  echo ""
  echo "Step 2: Running B_run_FIDASIM.sh"
  echo "--------------------------------------------"

  # Prior to runnign FIDASIM, delete all sink and birth HDF5 files:
  rm *sink*h5
  rm *birth*h5
  echo "Clearing sink*.h5 and birth*.h5 prior to running FIDASIM"

  $PREPROCESSOR_DIR/$PYTHON_PACKAGE/B_run_FIDASIM.sh
  if [ $? -ne 0 ]; then
      echo "Error: B_run_FIDASIM.sh failed"
      exit 1
  fi
fi

# ===================================================================
# Step 3: Save births and sinks into text file (.dat):
# ============================================================
# Using the files outputted by FIDASIM in STEP 2. we can now write the text files needed for CQL3DM:
if [[ "$FIDASIM_RUN_SRCS_TO_TXT" == "1" ]]; then
  echo ""
  echo "Step 3: Running C_save_sources.py"
  echo "--------------------------------------------"

  # Python debugging options:
  DEBUG_CMD="-m pdb "
  DEBUG_CMD=""

  python $DEBUG_CMD $PREPROCESSOR_DIR/$PYTHON_PACKAGE/C_save_sources.py
  if [ $? -ne 0 ]; then
      echo "Error: C_save_sources.py failed"
      exit 1
  fi
fi

echo ""
echo "All FIDASIM steps completed successfully!"
echo "================================================================================================================="
echo ""