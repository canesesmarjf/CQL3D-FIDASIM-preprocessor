#!/bin/bash
#
# -------------------------------------------------------------------------
# Script Name:  request_interactive_PERLMUTTER.sh
#
# Purpose:
#   Request an interactive session on NERSC Perlmutter using `salloc`.
#   This script is useful for quick debugging, compiling, testing code,
#   running short jobs interactively, or launching tools like Python or gdb.
#
# Usage:
#   1. Edit the ACCOUNT, TIME, CONSTRAINT, etc. fields below as needed.
#   2. Run this script from the terminal (not via sbatch):
#        ./request_interactive_PERLMUTTER.sh
#
# Notes:
#   - This script must be run on a NERSC login node (NERSC_HOST=perlmutter)
#   - It does NOT submit a batch job — it opens an interactive shell
#   - Logs are not automatically saved; use `tee` or `script` if needed
#
# Author: Juan F. Caneses‑Marin (jfcm)
# -------------------------------------------------------------------------

# Safety check: must be on NERSC Perlmutter
if [ "$NERSC_HOST" != "perlmutter" ]; then
    echo "This script is only intended for use on NERSC Perlmutter."
    echo "    Detected NERSC_HOST='$NERSC_HOST'"
    exit 1
fi

# Set job config
ACCOUNT="m77"       # <-- CHANGE THIS
TIME="00:30:00"       # e.g., 30 min
NODES=1
CONSTRAINT="cpu"      # or gpu
QOS="interactive"

# --- Launch interactive session ---
echo "   Requesting interactive session on Perlmutter..."
echo "   Account:    $ACCOUNT"
echo "   Time:       $TIME"
echo "   Nodes:      $NODES"
echo "   Constraint: $CONSTRAINT"
echo "   QOS:        $QOS"
echo

salloc -N $NODES -C $CONSTRAINT -q $QOS -t $TIME --account=$ACCOUNT
