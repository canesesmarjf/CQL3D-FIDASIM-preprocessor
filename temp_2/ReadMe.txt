
# ============================
CQL3D_to_FIDASIM_preprocessor
# ============================

This repository contains the python package "CQL3D_to_FIDASIM_preprocessor".
This package has the functionality required to take CQL3D files and convert them into a form usable by FIDASIM.
In addition, we provide examples on how to use this package to produce the input files and run the FIDASIM executable.

IMPORTANT:
==========
You will need to have FIDASIM installed in your computer to make use of its python libraries.
In addition, you need to specify the FIDASIM root directory in variable FIDASIM_DIR of the python script "Step_1_assemble_FIDASIM_inputs" to allow access to the FIDASIM python libraries

This repository contains the following directories and files:
====================================================
- ./Step_1_input/
- ./Step_1_env/
- ./Step_1_output/
- ./CQL3D_to_FIDASIM_preprocessor/
- ReadMe.txt
- Step_1_assemble_FIDASIM_inputs.py
- Step_2_run_FIDASIM.sh
- Step_3_save_birth_txt.py
- Step_4_WHAM_preview_results.m

Here the description of the directories:
====================================================
- ./Step_1_input/:
---------------------------------------------------
Directory containing all CQL3DM files needed to setup the FIDASIM input HDF5 files.
- cqlinput (fortran namelist)
- standard CQL3DM output file (netCDF file)
- EQDSK file
- f4d ion and electron distribution .nc files (netCDF files)
- preprocessor configuration file (fortran namelist)

 NOTE: The configuration file tells the preprocessor how to process the CQL3D files into the format needed to run FIDASIM. It also contains FIDASIM specific parameters such as the size and resolution of the interpolation grid and beam grid in FIDASIM.

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
Python script which uses the package "CQL3D_to_FIDASIM_preprocessor" to create the following HDF5 FIDASIM input files in the directory ./Step_1_output.

- WHAM_example_input.dat
- WHAM_example_geometry.h5
- WHAM_example_equilibrium.h5
- WHAM_example_distribution.h5

To define the preprocessor and FIDASIM input parameters, modify "preprocessor_config_1a.dat" file in ./Step_1_input

- Step_1_env:
----------------------------------------------------
Contains FIDASIM_env.yml and requirements.txt:
I have included these two files in case you need them to setup the environment

- Step_2_run_FIDASIM.sh
---------------------------------------------------
Shell scripts that make use of the newly created FIDASIM HDF5 files and runs FIDASIM in either MPI or openMP mode. You need to compile FIDASIM with either MPI or openMP flags but not both.

This shell script will run the FIDASIM executable and will use the output files created in ./Step_1_output.
To run this shell script, type the following into the command line:

  ./Step_2_run_FIDASIM [OPENMP or MPI] [DEBUG]

The arguments [] are optional and tell the operating system how to run FIDASIM.
The DEBUG mode will attempt to run FIDASIM using the FORGE LINARO debugger.
The OPENMP or MPI will run FIDASIM in parallel mode using the default number of threads/processes.
To specify number of threads/processes, modify the shell script.

If you do run FIDASIM, then expect the following output files created in ./Step_1_output:
- WHAM_example_neutrals.h5
- WHAM_example_birth.h5

How to use this package:
=========================
1 - ENVIRONMENT: Setup your environment so you can import all required packages (see Step_1_env)
    + Could use CONDA to setup virtual environment to run python

2 - PREPARE PREPROCESSOR: Open "Step_1_assemble_FIDASIM_inputs.py" and and configure the run by modifying the following variables:
    + FIDASIM_dir (path to FIDASIM repo in local computer)
    + file_name (path to the preprocessor configuration file)
    + plot_flag (flag to enable plotting profiles for diagnostics purposes)
    + include_f4d (flag to include or ignore the interpolation step for ion distribution function)

3 - RUN PREPROCESSOR: Run the python script. No input argument needed:

    python Step_1_assemble_FIDASIM_inputs.py

4 - TROUBLESHOOT: Check the command line as this will tell you if the process runs to completion and what errors it encountered.

5 - CHECK RESULTS: Check ./Step_1_outputs for the HDF5 files and .dat namelist file required for FIDASIM

6 - USE RESULTS: Run FIDASIM with newly created files. Use provided shell script:
    + OpenMP mode: ./Step_2_run_FIDASIM.sh OPENMP
    + MPI mode: ./Step_2_run_FIDASIM.sh MPI
    + OpenMP DEBUG mode: ./Step_2_run_FIDASIM.sh OPENMP DEBUG
    + MPI DEBUG mode: ./Step_2_run_FIDASIM.sh MPI DEBUG

7 - POSTPROCESS: Run Step_3_save_birth_txt.py to process FIDASIM output file containing the fast ion data and convert it in to a format readable by CQL3D. This process is automatically performed by "Step_2_run_FIDASIM.sh"

8 - (OPTIONAL) POSTPROCESS: preview output from the preprocessor (step 2) by plotting data with MATLAB
