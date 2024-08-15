import cql3d_to_fidasim_preprocessor as pp

# Add local FIDASIM directory:
FIDASIM_dir = "/home/jfcm/Repos/FIDASIM/"
pp.set_fidasim_dir(FIDASIM_dir)

# Read the configuration file which specifies how to run FIDASIM with CQL3D input files:
file_name = "./Step_1_input/preprocessor_config_1a.nml"
config = pp.read_preprocessor_config(file_name)

# Create FIDASIM input files using PREFIDA:
plot_flag = True
pp.create_fidasim_inputs_from_cql3dm(config, plot_flag)
print("End of script")
