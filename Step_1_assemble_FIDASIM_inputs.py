import os
import sys
import argparse
import cql3d_to_fidasim_preprocessor as pp

# Get command line arguments:
# ======================================================================================================================
description = "This script reads CQL3D files and processes them to make FIDASIM input files"
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--config_dir', type=str, help="Directory where the FIDASIM configuration file is located")
parser.add_argument('--plot', action='store_true', help="Enable plotting")  # Add the --plot flag
parser.add_argument('--fidasim_dir', type=str, help="Directory where the FIDASIM repo is located")
args = parser.parse_args()

# Add local FIDASIM directory:
if (args.fidasim_dir == None):
    args.fidasim_dir = os.getenv('FIDASIM_DIR','/home/jfcm/Repos/FIDASIM')
pp.set_fidasim_dir(args.fidasim_dir)

# Read the configuration file which specifies how to run FIDASIM with CQL3D input files:
# ======================================================================================================================
run_id = os.path.basename(os.path.normpath(args.config_dir))
config_file = args.config_dir + run_id + "_config.nml"
if not os.path.exists(config_file):
    print(f"Error: The file '{config_file}' does not exist.")
    sys.exit(1)  #
config = pp.construct_preprocessor_config(config_file)

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