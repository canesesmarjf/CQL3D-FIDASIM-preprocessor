#!/bin/bash
#
# Note: Source this script instead of executing it directly to ensure that Conda activation affects the parent shell.
#
# Usage: source script.sh [options]
#
# This script creates the CONDA environment "FIDASIM_env"
# This conda environment is required to run the CQL3DM-FIDASIM preprocessor.
#
# Run this script just after cloning the preprocessor repo.
# The correct conda environment is required to run the preprocessor python scripts.
# The file which species the environment needed to install the required conda environment is: environment.yml
# This file was created using the following:
#
# conda env export > environment.yml
#
# To create a new environment using this file use the following statement:
#
# conda env create -f environment.yml
#
# Once created, you can active the environment in the command line using:
#
# conda activate FIDASIM_env
#
# After that, all the preprocessor python scripts will work

# Function to display usage information:
# ====================================================================================
usage() {
    echo "Usage: source $0 [options]"
    echo ""
    echo "Options:"
    echo "  --activate, -a      Create the environment and activate it immediately if created successfully."
    echo "  --help, -h          Display this help message and exit."
    echo ""
    echo "Description:"
    echo "This script sets up the CONDA environment 'FIDASIM_env', required for running the CQL3DM-FIDASIM preprocessor."
    echo "If 'FIDASIM_env' already exists, it skips environment creation. The optional --activate flag activates"
    echo "the environment after creation or if it already exists."
    echo ""
    echo "Example Usage:"
    echo "  $0                    # Create the environment if it does not exist"
    echo "  $0 --activate or -a   # Create the environment if it does not exist, and activate it"
    echo "  $0 --help or -h       # Show this help message"
    exit 0
}

# Function to initialize Conda:
# ====================================================================================
# Works on systems like Perlmutter using module load or on local machines
initialize_conda() {
    # Check if module command is available (indicates module environment)
    if command -v module &> /dev/null; then
        # Load the appropriate Conda module (change if a specific version is needed)
        module load python  # This usually provides Conda
        echo "Loaded Conda via module load."
    else
        # Locate the Conda base and source conda.sh as a fallback
        CONDA_BASE=$(dirname $(dirname $(which conda)))
        CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

        if [[ -f "$CONDA_SH" ]]; then
            source "$CONDA_SH"
            echo "Conda environment has been sourced from $CONDA_SH"
        else
            echo "Error: Could not initialize Conda. Conda.sh not found and module load not available."
            exit 1
        fi
    fi
}

# Parse command line arguments:
# ====================================================================================
ACTIVATE_ENV=false
for arg in "$@"; do
    case $arg in
        --activate|-a)
            ACTIVATE_ENV=true
            shift
            ;;
        --help|-h)
            usage  # Show usage information and exit
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            ;;
    esac
done

# Initialize Conda to ensure it is available
# ====================================================================================
initialize_conda

# Verify Conda is now accessible
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Conda before proceeding."
    exit
fi

# Create environment from .yml file:
# ====================================================================================
# Check if the environment already exists
if conda env list | grep -q "^FIDASIM_env"; then
    echo "The environment 'FIDASIM_env' already exists. Skipping environment creation."
else
    # Create environment from environment.yml
    conda env create -f ./environment.yml
fi

# Optionally activate the environment if --activate was specified:
# ====================================================================================
if $ACTIVATE_ENV; then
    echo "Activating the environment 'FIDASIM_env'..."
    conda activate FIDASIM_env
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to activate 'FIDASIM_env'. Ensure Conda is set up properly."
        exit 1
    fi
    echo "'FIDASIM_env' is now active."
fi
