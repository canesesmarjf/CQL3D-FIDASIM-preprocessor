#!/bin/bash
# This script automates the running of FIDASIM using various command line arguments.
# We need to make sure that we match the parallel mode selected to how the source has built:
# with MPI, openMP and/or debug mode

# Default values for options
executable=""
input_file=""
parallel_mode="openmp"
num_threads=0
verbose=false
prompt=false
debug=false

# Function to print usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --input-file FILE           Path to the input file (required, must be .dat)"
    echo "  -e, --executable FILE           Path to the executable file (default: \$FIDASIM_DIR/fidasim)"
    echo "  -p, --parallel-mode MODE        Parallelization mode: 'openmp' or 'mpi' (default: openmp)"
    echo "  -n, --num-threads NUM           Number of threads/ranks for OpenMP/MPI (default: 14)"
    echo "  -v, --verbose                   Display the setup before running"
    echo "      --prompt                    Display the setup and ask for confirmation"
    echo "      --debug                     Run the executable in debug mode (using ddt --connect)"
    echo "      --help                      Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  $0 -i input.dat -e /path/to/executable -p mpi -n 16 --debug"
    echo "  $0 -i input.dat --parallel-mode openmp --num-threads 8 --verbose --debug"
    echo ""
    echo "If -e is not provided, the script will use the default executable \$FIDASIM_DIR/fidasim"
    exit 1
}

# Function to calculate half of available CPUs (rounded down)
calculate_num_threads() {
    total_cpus=$(nproc)  # On Linux, this will return the total number of CPUs
    num_threads=$((total_cpus / 2))  # Use integer division, automatically rounds down

    # Ensure at least 1 thread is used if there's only 1 CPU
    if [[ "$num_threads" -lt 1 ]]; then
        num_threads=1
    fi
}

# Function to display the setup
display_setup() {
    echo "Executable: $executable"
    echo "Input file: $input_file"
    echo "Parallelization mode: $parallel_mode"
    echo "Number of threads/ranks: $num_threads"
    echo "Debug mode: $debug"
}

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

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -e|--executable) executable="$2"; shift ;;
        -i|--input-file) input_file="$2"; shift ;;
        -p|--parallel-mode) parallel_mode="$2"; shift ;;
        -n|--num-threads) num_threads="$2"; shift ;;
        -v|--verbose) verbose=true ;;
        --prompt) prompt=true ;;
        --debug) debug=true ;;
        --help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# If number of threads is not manually set, calculate it
if [[ "$num_threads" -eq 0 ]]; then
    echo ""
    echo "NUM_THREADS: =================================="
    echo "num_threads not set..."
    echo "Using half of the system's maximum threads ..."
    calculate_num_threads  # Call function to calculate half of the available CPUs
    echo "Number of threads/ranks: $num_threads"
fi

# Check the state of parallel_mode
if [[ "$parallel_mode" != "openmp" && "$parallel_mode" != "mpi" ]]; then
    echo "Invalid mode: $parallel_mode. Please choose either 'openmp' or 'mpi'."
    exit 1
fi

# Check if input file is provided
if [[ -z "$input_file" ]]; then
    echo "Error: FIDASIM *.dat input file is required."
    usage
elif [[ "$input_file" != *.dat ]]; then
    echo "Error: Input file must be a FIDASIM .dat file."
    exit 1
fi

# Check if executable is provided or use the default from $FIDASIM_DIR
echo ""
echo "EXECUTABLE: =================================="
if [[ -z "$executable" ]]; then
    if [[ -z "$FIDASIM_DIR" ]]; then
        echo "Error: No executable provided and \$FIDASIM_DIR is not set."
        exit 1
    else
        executable="$FIDASIM_DIR/fidasim"
        echo "No executable provided. Using default from local repository: $executable"
        check_executable "$executable"
    fi
else
    check_executable "$executable"
fi

# If prompt is true, verbose will always be true:
if $prompt; then
    verbose=true
fi

# Verbose: Simply display the setup
if $verbose; then
    echo ""
    echo "Summary of inputs: =================================="
    display_setup
fi

# Prompt: ask for confirmation on inputs
if $prompt; then
    echo ""
    echo "Check inputs: =================================="
    read -p "Does this look correct? (Y/N): " confirmation
    if [[ "$confirmation" != "Y" ]]; then
        echo "Aborting the script."
        exit 1
    fi
fi

if $debug; then
  # Define the directory where the Forge binaries are located
  FORGE_DIR=/home/jfcm/linaro/forge/24.0.2/bin

  # Export FORGE_DIR to the PATH
  export PATH=$FORGE_DIR:$PATH
fi

# Execute the program based on the parallelization mode and debug flag:
echo ""
echo "Run mode: =================================="
if [[ "$parallel_mode" == "openmp" ]]; then
    if $debug; then
        echo "Running in DEBUG mode (reverse connect to FORGE)..."
        ddt --connect $executable $input_file $num_threads
    else
        echo "Running in OpenMP mode with $num_threads threads."
        $executable $input_file $num_threads
    fi
elif [[ "$parallel_mode" == "mpi" ]]; then
    if $debug; then
        echo "Running in DEBUG mode (reverse connect to FORGE)..."
        ddt --connect mpirun -n $num_threads $executable $input_file
    else
        echo "Running in MPI mode with $num_threads ranks."
        mpirun -np $num_threads $executable $input_file
    fi
fi

echo ""
echo "FIDASIM run completed... ==================="

exit 0
