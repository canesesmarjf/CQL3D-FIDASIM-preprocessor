#!/bin/bash

# USER INPUTS:
# ============================================================
# ======================= USER INPUTS ========================
# ============================================================

# Case to run:
RUN_ID="WHAM_example_2"
RUN_ID="WHAM_no_f4d"
RUN_DIR=./fidasim_files/$RUN_ID

# ENV variables:
FIDASIM_DIR=$FIDASIM_DIR
PREPROCESSOR_DIR=$PREPROCESSOR_DIR

# ============================================================
# ======================= USER INPUTS ========================
# ============================================================

# Step 1: Run preprocessor script to produce input files for FIDASIM:
# ============================================================
python $PREPROCESSOR_DIR/Step_1_assemble_FIDASIM_inputs.py --run-dir $RUN_DIR --fidasim-dir $FIDASIM_DIR --plot
if [ $? -ne 0 ]; then
    echo "Error: Step_1_assemble_FIDASIM_inputs.py failed"
    exit 1
fi

# Step 2: Run FIDASIM:
# ============================================================
# The intention here is that we only need to provide the name of run_ID or run_config file and then all input files are automatically selected.
# We need to make Step_1_assemble_FIDASIM_inputs also accpet command line arguments
# We might need to store config file and all fidasim related outouts in ./fidasim_files/output/run_ID/
# config.nml, run_ID_inputs.dat, *.h5, *_births_f1.text.
# So we would just guide the calculation by telling the coupler to cjoose the run_ID direcotry and then search there for the required inputs such as *.dat (FIDASIM) or *.nml file (cql3d_to_fidasim_preprocessor)

# Need to make this shell script accept only run directories and then autoamtically looks for a .dat file:
INPUT_FILE=$RUN_DIR/${RUN_ID}_inputs.dat
./Step_2_run_FIDASIM.sh -i $INPUT_FILE --parallel-mode openmp -n 14 -e $FIDASIM_DIR/fidasim --verbose
if [ $? -ne 0 ]; then
    echo "Error: Step_2_run_FIDASIM.sh failed"
    exit 1
fi

# Step 3: Save sources and sinks into text file (.dat):
# ============================================================
python $PREPROCESSOR_DIR/Step_3_save_sources_and_sinks.py --run-dir $RUN_DIR
if [ $? -ne 0 ]; then
    echo "Error: Step_3_save_sources_and_sinks.py failed"
    exit 1
fi

echo "All FIDASIM steps completed successfully!"