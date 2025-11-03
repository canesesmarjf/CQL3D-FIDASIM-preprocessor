#!/bin/bash

# Enable the use of conda:
source ../cql3d_to_fidasim_preprocessor/setup_conda.sh

# Remove existing FIDASIM_env:
#conda env remove -n FIDASIM_env

# Create conda env:
conda env create -f environment.yml

# To create new environment.yml, use the following two options:
# conda env export > environment.yml
# conda env export --no-builds > environment.yml
#
# The first option exports the full and exact environment which includes build info and OS specific details
