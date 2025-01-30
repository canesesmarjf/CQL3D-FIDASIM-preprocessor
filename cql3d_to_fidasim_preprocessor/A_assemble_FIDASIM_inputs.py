import os
import sys
import argparse
import preprocessor as pp

# Get command line arguments:
# ======================================================================================================================
description = "This script reads CQL3D files and processes them to make FIDASIM input files"
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--fida-run-dir', type=str, help="Path to the FIDASIM run directory. Ending in $run id")
parser.add_argument('--cql-run-dir', type=str, help="Path to the CQLD3D run directory. Ending in $run_id")
parser.add_argument('--plot', action='store_true', help="Enable plotting")  # Add the --plot flag
parser.add_argument('--fidasim-dir', type=str, help="Directory where the FIDASIM repo is located")
args = parser.parse_args()

# USER INPUTS:
# ============================================================
# ======================= USER INPUTS ========================
# ============================================================

# Check if no command line arguments were provided
if len(sys.argv) == 1:
    print("No command line arguments provided. Using internal values.")

    # User input:
    # run_id = "WHAM_low_ne_thermal"
    # run_id = "WHAM_wall_flux_cold_plasma"
    # run_id = "WHAM_low_ne_nonthermal"
    #
    # fidasim_run_dir = "./fidasim_files/" + run_id
    # cql3d_run_dir   = "./cql3d_files/" + run_id

    run_id = "WHAM_high_ne_nonthermal_multistep_cx"
    fidasim_run_dir = "../Step_1b_standalone_runs/" + run_id
    cql3d_run_dir   = "../Step_1b_standalone_runs/" + run_id

    args.fida_run_dir = fidasim_run_dir
    args.cql_run_dir = cql3d_run_dir
    args.fidasim_dir = None
    args.plot = True

# ============================================================
# ======================= USER INPUTS ========================
# ============================================================

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

print("")
print("End of script")
print("Completed A_assemble_FIDASIM_inputs.py")
print("--------------------------------------------")

# Notes:
# ==================
# At present, the f4d interpolation takes the longest to compute in the preprocessing
# To improve this we can (1) make it into a fotran subroutine or (2) find a more efficient way to perform the interpolation

# Future changes:
# Consider performing the preprocessing all in fortran