#!/bin/bash
#
# clean_dir.sh
#
# Deletes output files from FIDASIM, CQL3D, and SLURM jobs.
#
# Usage:
#   ./clean_dir.sh           # clean all (*.h5 *.dat *.png *.nc *.ps *.out *.err)
#   ./clean_dir.sh fidasim   # clean only FIDASIM files (*.h5 *.dat *.png)
#   ./clean_dir.sh cql3d     # clean only CQL3D files (*.nc *.ps)
#   ./clean_dir.sh logs      # clean only SLURM log files (*.out *.err)
#   ./clean_dir.sh --help    # show this message
#

# --- Help message ---
if [[ "$1" == "--help" ]]; then
    echo "Usage:"
    echo "  ./clean_dir.sh           # clean all (*.h5 *.dat *.png *.nc *.ps *.out *.err)"
    echo "  ./clean_dir.sh fidasim   # clean only FIDASIM files (*.h5 *.dat *.png)"
    echo "  ./clean_dir.sh cql3d     # clean only CQL3D files (*.nc *.ps)"
    echo "  ./clean_dir.sh logs      # clean only SLURM log files (*.out *.err)"
    echo "  ./clean_dir.sh --help    # show this message"
    exit 0
fi

# --- Defaults ---
DO_FIDASIM=false
DO_CQL3D=false
DO_LOGS=false

if [ $# -eq 0 ]; then
    DO_FIDASIM=true
    DO_CQL3D=true
    DO_LOGS=true
else
    for arg in "$@"; do
        case "$arg" in
            fidasim) DO_FIDASIM=true ;;
            cql3d)   DO_CQL3D=true ;;
            logs)    DO_LOGS=true ;;
            *) echo "Unknown option: $arg"; exit 1 ;;
        esac
    done
fi

$DO_FIDASIM && echo "Deleting FIDASIM files..." && rm -f *.h5 *.dat *.png ./figures/*.png
$DO_CQL3D   && echo "Deleting CQL3D files..."   && rm -f *.nc *.ps
$DO_LOGS    && echo "Deleting SLURM logs..."    && rm -f *.out *.err ./logs/*.out ./logs/*.err

echo "Done."