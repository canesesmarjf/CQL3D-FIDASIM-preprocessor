#!/bin/bash

if [[ "$NERSC_HOST" == "perlmutter" ]]; then
  # Load required modules:
  module load forge
  module load cray-hdf5
  module load cray-netcdf
  module load conda/Miniforge3-24.7.1-0
  module load PrgEnv-gnu
fi