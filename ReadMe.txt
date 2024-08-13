The objective of this directory is compute the FIDASIM input files directly from the CQL3DM files

As a secondary objective, we use these newly created input files to run FIDASIM in any of the run formats: OpenMP, MPI, debug MPI and debug single thread.

The output files created by FIDASIM will be stored in the output_dir variable defined in the following dict:

# Define path to output directory:
# (This is where you want the FIDASIM HDF5 files to be written to)
user_input['output_path'] = "./Step_1_output/"

IMPORTANT:
==========
You will need to have FIDASIM installed in your computer to make use of its python libraries and modify the "FIDASIM_DIR" in the python script to allow access to these libraries

This directory contains:
=======================
- ./Step_1_input/:
---------------------------------------------------
Directory containing all CQL3DM files
Contains the files from CQL3DM needed to setup the FIDASIM input HDF5 files.
- cqlinput
- standard CQL3DM .nc output file
- EQDSK file
- f4d ion and electron distribution .nc files

- ./Step_1_output:
---------------------------------------------------
Directory containing all the FIDASIM HDF5 files produced by python script processing CQL3DM inputs

Python script should create the following files:
- WHAM_example_input.dat
- WHAM_example_geometry.h5
- WHAM_example_equilibrium.h5
- WHAM_example_distribution.h5

if plot_flag == True, then 11 figures will be produced so you can inspect the profiles

- Step_1_assemble_FIDASIM_inputs.py:
---------------------------------------------------
Python script to produce HDF5 FIDASIM input files.
To define FIDASIM input parameters, modify "user_input" dictionary in the main() function

- Step_1_env:
----------------------------------------------------
Contains FIDASIM_env.yml and requirements.txt:
I have included these two files in case you need them to setup the environment

- Step_2_run_*.sh
---------------------------------------------------
Shell scripts that make use of the newly created FIDASIM HDF5 files and runs FIDASIM in either MPI or openMP mode. You need to compile FIDASIM with MPI or openMP flags but not both.

If you do run FIDASIM, then expect the following output files:
- WHAM_example_neutrals.h5
- WHAM_example_birth.h5

How to run python script:
=========================
1 - Setup your environment so you can import all required packages (see Step_1_env)
2 - Go to the function "main()" in the python script and configure the run by modifying the following variables/dictionaries:
  + user_input (dictionary)
  + plot_flag (variable) (plots profiles for diagnostics purposes)
3 - run the python script. No input argument needed
4 - Check the command line as this will tell you if the process run to completion and what errors it encountered.
5 - Check ./Step_1_outputs for the HDF5 files and .dat namelist file required for FIDASIM
