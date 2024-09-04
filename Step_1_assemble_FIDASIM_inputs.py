import cql3d_to_fidasim_preprocessor as pp

# Add local FIDASIM directory:
FIDASIM_dir = "/home/jfcm/Repos/FIDASIM/"
pp.set_fidasim_dir(FIDASIM_dir)

# Read the configuration file which specifies how to run FIDASIM with CQL3D input files:
file_name = "./Step_1_input/preprocessor_config_1a.nml"
config = pp.read_preprocessor_config(file_name)

# Create FIDASIM input files using PREFIDA:
plot_flag = True
include_f4d = True
pp.create_fidasim_inputs_from_cql3dm(config, plot_flag, include_f4d)

print("End of script")

# Notes:
# ==================
# We have added "include_f4d" to allow skipping the interpolation of the distribution function.
# This has been done because at present this is the step that takes the longest to compute in the preprocessing
# To improve this we can (1) make it into a fotran subroutine or (2) find a more efficient way to perform the interpolation

# Future changes:
# Remove the plotting scripts from the main process and create a function that reads the FIDASIM input files and plots results
# Consider performing the preprocessing all in fortran