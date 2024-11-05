#!/bin/bash

# Setup CONDA python environment:
# ===========================================================
# This is required to run the FIDASIM-CQL3DM preprocessor

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate FIDASIM_env

# Define run directories:
# ===========================================================
export RUN_ID="WHAM_no_f4d"
export FIDASIM_RUN_DIR="$PWD/fidasim_files/$RUN_ID"
export CQL3D_RUN_DIR="$PWD/cql3d_files/$RUN_ID"
export START_DIR=$PWD

# Setup environmental variables:
# ===========================================================
# Define the directory where the Forge binaries are located
FORGE_DIR=/home/jfcm/linaro/forge/24.0.2/bin

# Export FORGE_DIR to the PATH
export PATH=$FORGE_DIR:$PATH

# Number of processors:
NUM_PROCS=11

# Executable:
EXECUTABLE=./xcql3dm_mpi.gfortran64_DEBUG

# Run CQL3DM:
# ===========================================================
cd $CQL3D_RUN_DIR
#time ddt --connect mpirun -n $NUM_PROCS $CQL3DM_DIR/xcql3dm_mpi.gfortran64_DEBUG > log.txt
time ddt --connect mpirun -n $NUM_PROCS $CQL3DM_DIR/xcql3dm_mpi.gfortran64_DEBUG
cd $START_DIR
