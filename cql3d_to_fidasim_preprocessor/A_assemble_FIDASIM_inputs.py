import os
import preprocessor as pp

# USER INPUTS:
# ============================================================
# ======================= USER INPUTS ========================
# ============================================================

# Check if env vars have been set:
if os.getenv("RUN_ID") is None:
    print("")
    print("==========================================================")
    print("No command line arguments provided. Using internal values.")
    print("==========================================================")
    print("")

    # User input:
    run_id = "WHAM_test"
    os.environ["RUN_ID"] = run_id
    os.environ["RUN_DIR"] = "/home/jfcm/Repos/CQL3D-FIDASIM-preprocessor/Step_1b_standalone_runs/" + run_id
    os.environ["FIDASIM_DIR"] = "/home/jfcm/Repos/FIDASIM"
    os.environ["PREPROCESSOR_PLOT_CREATE"] = "1"
    os.environ["PREPROCESSOR_PLOT_SHOW"] = "1"
    os.environ["PREPROCESSOR_PLOT_SAVE"] = "1"

# ============================================================
# ======================= USER INPUTS ========================
# ============================================================

# Add local FIDASIM directory:
pp.set_fidasim_dir(os.getenv('FIDASIM_DIR'))

# Read the configuration file which specifies how to run FIDASIM with CQL3D input files:
# ======================================================================================================================
run_dir = os.getenv('RUN_DIR')
config = pp.construct_preprocessor_config(run_dir)

# Create FIDASIM input files using PREFIDA:
# ======================================================================================================================
plot_flag = os.getenv("PREPROCESSOR_PLOT_CREATE") == "1"
pp.construct_fidasim_inputs_from_cql3d(config, plot_flag)

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