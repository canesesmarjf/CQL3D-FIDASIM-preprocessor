#!/bin/bash
if [[ "$NERSC_HOST" == "perlmutter" ]]; then
  echo "Running on Perlmutter"
  module load conda/Miniforge3-24.7.1-0
else
  echo "Running from" $HOSTNAME
  source ~/miniconda3/etc/profile.d/conda.sh
fi
