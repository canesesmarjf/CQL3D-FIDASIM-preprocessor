#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --mail-user=caneses@compxco.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ================= START USER INPUTS: ======================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Computational resources:
FIDASIM_NUM_THREADS=16

# Select FIDASIM processes to run in the workflow:
FIDASIM_RUN_PREPROCESSOR=0
FIDASIM_RUN_EXEC=1
FIDASIM_RUN_SRCS_TO_TXT=1

# Output options:
PREPROCESSOR_PLOT_CREATE=1 # Create figures
PREPROCESSOR_PLOT_SHOW=0 # Show figures on screen
PREPROCESSOR_PLOT_SAVE=1 # Save figures

# Environment:
FIDASIM_DIR="/global/homes/j/jfcm/myRepos/FIDASIM"
PREPROCESSOR_DIR="/global/homes/j/jfcm/myRepos/CQL3D-FIDASIM-preprocessor"

# Executable:
FIDASIM_EXECUTABLE=fidasim

# Debug options:
FIDASIM_DEBUG=0
FORGE_DIR="/home/jfcm/linaro/forge/24.0.2/bin"

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==================== END USER INPUTS: =====================
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Launch FIDASIM workflow:
python_package="cql3d_to_fidasim_preprocessor/"
source $PREPROCESSOR_DIR/$python_package"setup_perlmutter.sh"
source $PREPROCESSOR_DIR/$python_package"setup_conda.sh"
source $PREPROCESSOR_DIR/$python_package"activate_conda_env.sh"
source $PREPROCESSOR_DIR/$python_package"export_variables.sh"
source $PREPROCESSOR_DIR/$python_package"Launch_FIDASIM_workflow.sh"
