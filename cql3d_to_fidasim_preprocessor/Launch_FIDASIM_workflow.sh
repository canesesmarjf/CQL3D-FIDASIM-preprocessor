#!/bin/bash

# Check status of required environmental variables:
# ==============================================================
if [ -z "$RUN_ID" ]; then
  echo "error in Launch_FIDASIM_workflow.sh: RUN_ID has not been set"
  echo ""
  exit 1
fi
if [ -z "$FIDASIM_RUN_DIR" ]; then
  echo "error in Launch_FIDASIM_workflow.sh: FIDASIM_RUN_DIR has not been set"
  exit 1
fi
if [ -z "$CQL3D_RUN_DIR" ]; then
  echo "error in Launch_FIDASIM_workflow.sh: CQL3D_RUN_DIR has not been set"
  exit 1
fi

echo ""
echo "Starting Launch_FIDASIM_workflow.sh:"
echo "================================================================================================================="
echo ""

# ============================================================
# ============================================================
# In what follows, we have four steps:
# ============================================================
# ============================================================
# STEP 0- Set the environment needed to run preprocessor
# STEP 1- Run preprocessor script to produce input files for FIDASIM
# STEP 2- Run FIDASIM
# STEP 3- Save sources and sinks in text files for CQL3DM

# ===================================================================
# STEP 0: SET THE ENVIRONMENT
# ===================================================================

# Setup CONDA python environment needed to run the FIDASIM-CQL3DM preprocessor:
# Initialize Conda
# >>> [JFCM, 2025-10-21] >>>
#source ~/miniconda3/etc/profile.d/conda.sh
# Activate the Conda environment
#conda activate FIDASIM_env
# <<< [JFCM, 2025-10-21] <<<

# When running FIDASIM in standalone mode, check for the following special variables:
# FIDASIM_RUN_PREPROCESSOR
# FIDASIM_RUN_EXEC
# FIDASIM_SRCS_TO_TXT
# FIDASIM_DEBUG

# If the above env vars are not defined, make them 1 by default, except the debug mode:
if [[ -z "$FIDASIM_RUN_PREPROCESSOR" ]]; then
    export FIDASIM_RUN_PREPROCESSOR=1
    echo "FIDASIM_RUN_PREPROCESSOR is not set. Setting to default: 1"
fi
if [[ -z "$FIDASIM_RUN_EXEC" ]]; then
    export FIDASIM_RUN_EXEC=1
    echo "FIDASIM_RUN_EXEC is not set. Setting to default: 1"
fi
if [[ -z "$FIDASIM_RUN_SRCS_TO_TXT" ]]; then
    export FIDASIM_RUN_SRCS_TO_TXT=1
    echo "FIDASIM_RUN_SRCS_TO_TXT is not set. Setting to default: 1"
fi
if [[ -z "$FIDASIM_DEBUG" ]]; then
    export FIDASIM_DEBUG=0
    echo "FIDASIM_DEBUG is not set. Setting to default: 0"
fi

# Print environmental variables::
[ -z "$RUN_ID" ] && echo "RUN_ID: not set" || echo "RUN_ID: $RUN_ID"
[ -z "$FIDASIM_RUN_DIR" ] && echo "FIDASIM_RUN_DIR: not set" || echo "FIDASIM_RUN_DIR: $FIDASIM_RUN_DIR"
[ -z "$CQL3D_RUN_DIR" ] && echo "CQL3D_RUN_DIR: not set" || echo "CQL3D_RUN_DIR: $CQL3D_RUN_DIR"

# ===================================================================
# STEP 1: Run preprocessor script to produce input files for FIDASIM:
# ===================================================================
# Step 1 requires the following:
# 1- [file] <RUN_ID>_config.nml (FIDASIM run configuration file)
# 2- [file] <RUN_ID>_cql_config.nml (CQL3D run configuration file)
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

  if [[ "$PREPROCESSOR_PLOT" == "1" ]]; then
      PLOT_FLAG="--plot"
  else
      PLOT_FLAG=""
  fi

  STEP_1="python $PREPROCESSOR_DIR/$PYTHON_PACKAGE/A_assemble_FIDASIM_inputs.py"
  $STEP_1 --fida-run-dir $FIDASIM_RUN_DIR --cql-run-dir $CQL3D_RUN_DIR --fidasim-dir $FIDASIM_DIR $PLOT_FLAG
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

  if [[ "$FIDASIM_DEBUG" == "1" ]]; then
      DEBUG_FLAG="--debug"
      echo ""
      echo "========================="
      echo "FIDASIM debugging enabled"
      echo "========================="
  else
      DEBUG_FLAG=""
  fi

  STEP_2="$PREPROCESSOR_DIR/$PYTHON_PACKAGE/B_run_FIDASIM.sh"
  $STEP_2 --fida-run-dir $FIDASIM_RUN_DIR --parallel-mode openmp -n $NUM_THREADS --executable $FIDASIM_DIR/fidasim $VERBOSE_FLAG $DEBUG_FLAG
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
  echo "Step 2: Running C_save_sources.py"
  echo "--------------------------------------------"
  STEP_3="python $PREPROCESSOR_DIR/$PYTHON_PACKAGE/C_save_sources.py"
  $STEP_3 --fida-run-dir $FIDASIM_RUN_DIR --cql-run-dir $CQL3D_RUN_DIR
  if [ $? -ne 0 ]; then
      echo "Error: C_save_sources.py failed"
      exit 1
  fi
fi

echo ""
echo "All FIDASIM steps completed successfully!"
echo "================================================================================================================="
echo ""