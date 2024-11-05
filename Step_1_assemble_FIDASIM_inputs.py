import os
import sys
import argparse
import cql3d_to_fidasim_preprocessor as pp

# Get command line arguments:
# ======================================================================================================================
description = "This script reads CQL3D files and processes them to make FIDASIM input files"
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--fida-run-dir', type=str, help="Path to the FIDASIM run directory. Ending in $run id")
parser.add_argument('--cql-run-dir', type=str, help="Path to the CQLD3D run directory. Ending in $run_id")

# parser.add_argument('--run-dir', type=str, help="Directory where the FIDASIM configuration file is located")

parser.add_argument('--plot', action='store_true', help="Enable plotting")  # Add the --plot flag
parser.add_argument('--fidasim-dir', type=str, help="Directory where the FIDASIM repo is located")
args = parser.parse_args()

# Add local FIDASIM directory:
if (args.fidasim_dir == None):
    args.fidasim_dir = os.getenv('FIDASIM_DIR','/home/jfcm/Repos/FIDASIM')
pp.set_fidasim_dir(args.fidasim_dir.rstrip('/'))

# Read the configuration file which specifies how to run FIDASIM with CQL3D input files:
# ======================================================================================================================
fida_run_dir = args.fida_run_dir.rstrip('/') + "/"
cql_run_dir  = args.cql_run_dir.rstrip('/') + "/"
config = pp.construct_preprocessor_config(fida_run_dir, cql_run_dir)

# Create FIDASIM input files using PREFIDA:
# ======================================================================================================================
pp.construct_fidasim_inputs_from_cql3d(config, args.plot)

print("End of script")

# Notes:
# ==================
# At present, the f4d interpolation takes the longest to compute in the preprocessing
# To improve this we can (1) make it into a fotran subroutine or (2) find a more efficient way to perform the interpolation

# Future changes:
# Consider performing the preprocessing all in fortran