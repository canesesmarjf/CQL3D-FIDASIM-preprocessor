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
export RUN_ID="WHAM_example_1"
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
NUM_PROCS=12 # Number of MPI processes for CQL3D
export NUM_THREADS=14 # Number of theads for FIDASIM

# Initialize run and time it:
cd $CQL3D_RUN_DIR
#time mpirun -n $NUM_PROCS $CQL3DM_DIR/xcql3dm_mpi.gfortran64 > log.txt
time mpirun -n $NUM_PROCS $CQL3DM_DIR/xcql3dm_mpi.gfortran64
cd $START_DIR
