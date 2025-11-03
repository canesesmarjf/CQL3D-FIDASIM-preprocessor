  #!/bin/bash
  #
  # -------------------------------------------------------------------------
  # Script Name:  submit_PERLMUTTER.sh
  #
  # Purpose:
  #   Submit a SLURM batch job on NERSC Perlmutter using either run_CQL3D.sh
  #   or run_FIDASIM.sh, depending on which one is present in the current
  #   working directory. Automatically sets the SLURM job name based on the
  #   directory name and routes logs to the ./logs folder.
  #
  # Usage:
  #   ./submit_PERLMUTTER.sh
  #
  # Notes:
  #   - Must be run on a Perlmutter login node (NERSC_HOST=perlmutter)
  #   - Only one of run_CQL3D.sh or run_FIDASIM.sh must be present
  #   - If both exist, the script will abort with a warning
  #   - SLURM logs are saved to logs/<jobname>-<jobid>.out and .err
  #
  # Author: Juan F. Caneses‑Marin (jfcm)
  # -------------------------------------------------------------------------
  
  # --- Safety check: must be on NERSC Perlmutter ---
  if [ "$NERSC_HOST" != "perlmutter" ]; then
      echo "This script is only intended for use on NERSC Perlmutter."
      echo "    Detected NERSC_HOST='$NERSC_HOST'"
      exit 1
  fi

  # --- Accept only one valid job script ---
  if [ -f run_CQL3D.sh ] && [ ! -f run_FIDASIM.sh ]; then
      RUN_SCRIPT="run_CQL3D.sh"
  elif [ -f run_FIDASIM.sh ] && [ ! -f run_CQL3D.sh ]; then
      RUN_SCRIPT="run_FIDASIM.sh"
  elif [ -f run_CQL3D.sh ] && [ -f run_FIDASIM.sh ]; then
      echo "Both run_CQL3D.sh and run_FIDASIM.sh exist. Please keep only one."
      exit 1
  else
      echo "No valid job script found. Expected run_CQL3D.sh or run_FIDASIM.sh."
      exit 1
  fi

  # --- Set job name based on directory ---
  JOB_NAME=$(basename "$PWD")

  # --- Create logs folder if needed ---
  mkdir -p logs

  # --- Print info ---
  echo "Submitting job using: $RUN_SCRIPT"
  echo "From directory:       $PWD"
  echo "Job name:             $JOB_NAME"

  # --- Submit with sbatch ---
  sbatch \
    --job-name="$JOB_NAME" \
    --output=logs/%x-%j.out \
    --error=logs/%x-%j.err \
    "$RUN_SCRIPT"
