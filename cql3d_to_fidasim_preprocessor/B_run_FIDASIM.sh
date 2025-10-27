#!/bin/bash

# Check if a file is executable and not a directory
check_executable() {
    if [[ -d "$1" ]]; then
        echo "Error: $1 is a directory, not an executable file."
        exit 1
    elif [[ ! -x "$1" ]]; then
        echo "Error: $1 is not a valid executable file."
        exit 1
    else
        echo "$1 executable file is valid"
    fi
}

# START OF PROGRAM:
# ======================================================================================================================

# Check if input file is provided
input_file="${RUN_DIR}/${RUN_ID}_inputs.dat"
if [[ -z "$input_file" ]]; then
    echo "Error: FIDASIM *.dat input file not found."
    usage
    exit 1
fi

# Check if executable is provided or use the default from $FIDASIM_DIR
echo ""
echo "EXECUTABLE: =================================="
executable=$FIDASIM_DIR/$FIDASIM_EXECUTABLE
check_executable "$executable"

echo ""
echo "Run mode: =================================="
RUN_CMD="$executable $input_file $FIDASIM_NUM_THREADS"
DEBUG_CMD="ddt --connect --openmp-threads=$FIDASIM_NUM_THREADS"

if [[ "${FIDASIM_DEBUG:-0}" -eq 1 ]]; then
    echo "Running in DEBUG mode (reverse connect to FORGE)..."
    $DEBUG_CMD $RUN_CMD &
    sleep 3
    $FORGE_DIR/forge
else
    echo "Running in OpenMP mode with $FIDASIM_NUM_THREADS threads."
    $RUN_CMD
fi

# Program fail check
if [ $? -ne 0 ]; then
    echo "Error in Step_2_run_FIDASIM.sh: Program execution failed with exit status $?"
    exit 1
fi

echo ""
echo "B_run_FIDASIM.sh  completed"
echo "============================"

exit 0