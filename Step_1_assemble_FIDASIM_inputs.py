import cql3d_to_fidasim_preprocessor as pp

# Add local FIDASIM directory:
FIDASIM_dir = "/home/jfcm/Repos/FIDASIM/"
pp.set_fidasim_dir(FIDASIM_dir)

# Select scenario to run:
scenario = "a"
match scenario:
    case "a" | 'a': # WHAM case
        file_name = "./Step_1_input/preprocessor_config_1a.nml"
        include_f4d = True
        plasma_from_cqlinput = False
        plot_flag = True
    case "b" | 'b': # ITER wall source case
        file_name = "./Step_1b_input/preprocessor_config_1b.nml"
        include_f4d = False
        plasma_from_cqlinput = True
        plot_flag = True
    case _:
        print("Error: Incorrect scenario selected")
        exit(0)

# Read the configuration file which specifies how to run FIDASIM with CQL3D input files:
config = pp.construct_preprocessor_config(file_name)

# Create FIDASIM input files using PREFIDA:
pp.construct_fidasim_inputs_from_cql3d(config, plot_flag, include_f4d, plasma_from_cqlinput)

print("End of script")

# Notes:
# ==================
# We have added "include_f4d" to allow skipping the interpolation of the distribution function.
# This has been done because at present this is the step that takes the longest to compute in the preprocessing
# To improve this we can (1) make it into a fotran subroutine or (2) find a more efficient way to perform the interpolation

# Future changes:
# Consider performing the preprocessing all in fortran