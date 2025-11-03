#!/bin/bash
if [[ "$NERSC_HOST" == "perlmutter" ]]; then
  # Load required modules:
  module load forge
  module load cray-hdf5
  module load cray-netcdf
  module load PrgEnv-gnu
fi
