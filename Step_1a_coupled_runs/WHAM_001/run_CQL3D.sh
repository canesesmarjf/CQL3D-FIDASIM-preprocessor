#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --mail-user=caneses@compxco.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=12

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ================= START USER INPUTS: ======================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Computational resources:
CQL3D_NUM_PROCS=10
FIDASIM_NUM_THREADS=12

# Output options:
CLI_TO_TXT=0 # Record terminal output to text file
PREPROCESSOR_PLOT_CREATE=1 # Create figures
PREPROCESSOR_PLOT_SHOW=1 # Show figures on screen
PREPROCESSOR_PLOT_SAVE=1 # Save figures

# Environment:
CQL3DM_DIR="$HOME/Repos/CQL3DM/2025_09_20/src"
FIDASIM_DIR="$HOME/Repos/FIDASIM"
PREPROCESSOR_DIR="$HOME/Repos/CQL3D-FIDASIM-preprocessor"

# Executable:
CQL3D_EXECUTABLE=xcql3dm_mpi.gfortran64
#CQL3D_EXECUTABLE=xcql3dm_mpi.gfortran64_DEBUG
#CQL3D_EXECUTABLE=xcql3d_mpi.perl
FIDASIM_EXECUTABLE=fidasim

# Debug options:
CQL3D_DEBUG=0
FORGE_DIR="/global/common/software/nersc9/forge/24.0.5/bin"

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==================== END USER INPUTS: =====================
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Launch CQL3D:
python_package="cql3d_to_fidasim_preprocessor/"
source $PREPROCESSOR_DIR/$python_package"setup_perlmutter.sh"
source $PREPROCESSOR_DIR/$python_package"setup_conda.sh"
source $PREPROCESSOR_DIR/$python_package"activate_conda_env.sh"
source $PREPROCESSOR_DIR/$python_package"export_variables.sh"
source $PREPROCESSOR_DIR/$python_package"Launch_CQL3D.sh"
