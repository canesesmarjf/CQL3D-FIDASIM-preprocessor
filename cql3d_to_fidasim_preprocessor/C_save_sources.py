# In this script, we read BOTH the birth.h5 and sink.h5 files produced by FIDASIM and create a .dat text file that can be read back into CQL3D
# In later stages, this script will need to read ALL birth and sink files and assemble a single .dat file

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import scipy.constants as consts
import h5py
import os
import sys
import argparse
import f90nml
import warnings
import socket
import time
from datetime import datetime

# ======================================================================================================================
# Get command line arguments:
# ======================================================================================================================
description = "This script reads CQL3D files and processes them to make FIDASIM input files"
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--fida-run-dir', type=str, help="Directory where the FIDASIM output HDF5 files are located")
parser.add_argument('--cql-run-dir', type=str, help="Directory where the CQL3D run files are located")

args = parser.parse_args()

# Input variables:
fida_run_dir = args.fida_run_dir.rstrip('/')
run_id = os.path.basename(os.path.normpath(fida_run_dir))

# ======================================================================================================================
# Define functions:
# ======================================================================================================================
def print_hdf5_structure(name, obj):
    print(name)
    for key, value in obj.attrs.items():
        print(f"  Attribute {key}: {value}")

def extract_hdf5_source_data(file_path):
    """
    Extracts data from an HDF5 file and returns it as a dictionary.
    Handles files with either 'n_birth' or 'n_sink' key.

    Parameters:
        file_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary containing the extracted data.
    """
    data = {}
    with h5py.File(file_path, 'r') as file:
        # Check which key is available and extract the corresponding data
        if 'n_birth' in file:
            data['n_type'] = file['n_birth'][()]  # Use 'n_type' as a common key
            data['type'] = 'birth'  # Add a type indicator
        elif 'n_sink' in file:
            data['n_type'] = file['n_sink'][()]
            data['type'] = 'sink'  # Add a type indicator
        else:
            raise KeyError("Neither 'n_birth' nor 'n_sink' found in the file.")

        # Common data keys
        data['energy'] = file['energy'][:] * 1e3  # [eV]
        if data['type'] == 'sink':
            data['weight'] = -file['weight'][:]  # [ions/s]
            data['denf4d'] = file['denf4d'][:] # [ions/cm^3 (s/cm)^3]
            data['denf4d_per_marker'] = file['denf4d_per_marker'][:] # [ions/cm^3 (s/cm)^3]
        else:
            data['weight'] = +file['weight'][:]  # [ions/s]
        data['ri'] = file['ri'][:]  # [cm]
        data['vi'] = file['vi'][:]  # [cm/s]

    return data

def list_files_with_extension(directory, extension):
    """
    Lists all files in the given directory with the specified extension.

    Parameters:
        directory (str): The target directory to search.
        extension (str): The file extension to filter by (e.g., ".h5").

    Returns:
        list: A list of filenames with the specified extension.
    """
    return [file for file in os.listdir(directory) if file.endswith(extension)]

def sorting_key(file_name):
    # Check for "birth.h5" or "birth_0.h5" and prioritize them
    if file_name.endswith("birth.h5") or file_name.endswith("birth_0.h5"):
        return 0  # Assign the highest priority
    elif file_name.endswith("sink.h5") or file_name.endswith("sink_0.h5"):
        return 0  # Also highest priority
    else:
        # Extract the numeric part (after "birth_" or "sink_") or assign a large number
        parts = file_name.split("_")
        if len(parts) > 2 and parts[2].startswith("birth") and parts[2][5:-3].isdigit():
            return int(parts[2][5:-3])
        elif len(parts) > 2 and parts[2].startswith("sink") and parts[2][4:-3].isdigit():
            return int(parts[2][4:-3])
        return int(parts[-1].split(".")[0]) if parts[-1].split(".")[0].isdigit() else float('inf')

def find_source_files(type,directory):
    """
    Finds source files of the given type ('sink' or 'birth') in the specified directory.

    Parameters:
        type (str): The type of files to search for ('sink' or 'birth').
        directory (str): The directory to search in.

    Returns:
        list: A list of file paths matching the criteria.

    Exits:
        Prints an error message and aborts if an invalid type is provided.
    """
    # List to store matching file paths
    source_files = []

    # Choose type and pattern to match:
    if type == 'sink':
        pattern = '_sink'
    elif type == 'birth':
        pattern = '_birth'
    else:
        print("Error with file type: use 'sink' or 'birth' type ...")
        sys.exit(1)  # Abort the program with a non-zero exit code

    # Get all HDF5 files in directory:
    h5_file_list = list_files_with_extension(directory, ".h5")

    # Get list of source files of type 'type' in directory:
    h5_source_filename = []
    for file_name in h5_file_list:
        if type in file_name:
            h5_source_filename.append(os.path.join(directory, file_name))

    # Sort the file names
    sorted_files = sorted(h5_source_filename, key=sorting_key)

    return sorted_files

def process_source_files(file_type, directory):
    """
    Finds and processes HDF5 source files of the given type in the specified directory.

    Parameters:
        file_type (str): The type of files to process ('sink' or 'birth').
        directory (str): The directory to search in.

    Returns:
        list: A list of dictionaries containing the processed data from each file.
    """
    # Find all source files of the specified type
    source_files = find_source_files(file_type, directory)

    # Extract data from each file
    source_data = []
    for file_path in source_files:
        source_data.append(extract_hdf5_source_data(file_path))

    # Compute power and flux term of source:
    for ii in range(len(source_data)):
        source_data[ii]["power"] = np.sum(source_data[ii]['weight']*source_data[ii]['energy']*consts.e) # [W]
        source_data[ii]["flux"] = np.sum(source_data[ii]['weight']) # [p/s]

    return source_data

def process_particle_data(data_list):
    """
    Processes particle data from a list of dictionaries to compute positions and velocities.

    Parameters:
        data_list (list): List of dictionaries containing 'ri', 'vi', 'weight', and optionally 'denf4d', 'denf4d_per_marker' keys.

    Returns:
        tuple: Arrays (X, Y, Z, vx, vy, vz, weights) with particle positions, velocities, weight and energy.
    """
    # Initialize empty lists to collect data
    X, Y, Z = [], [], []
    vx, vy, vz = [], [], []
    weight = []
    energy = []
    denf4d = []
    denf4d_per_marker = []

    # Iterate over each dictionary in the data list
    for data in data_list:
        # Extract positions
        ri = data['ri']  # Shape (N, 3), where columns are [R, Z, Phi]
        R = ri[:, 0]
        Z_data = ri[:, 1]
        Phi = ri[:, 2]

        # Convert to Cartesian coordinates
        X.extend(R * np.cos(Phi))  # X = R * cos(Phi)
        Y.extend(R * np.sin(Phi))  # Y = R * sin(Phi)
        Z.extend(Z_data)           # Z remains the same

        # Extract velocities
        vi = data['vi']  # Shape (N, 3), where columns are [vr, vz, vphi]
        vr = vi[:, 0]
        vz_data = vi[:, 1]
        vphi = vi[:, 2]

        # Compute perpendicular velocity (optional if needed later)
        v_perp = np.sqrt(vr**2 + vphi**2)

        # Convert velocities to Cartesian coordinates
        vx.extend(vr * np.cos(Phi) - vphi * np.sin(Phi))  # vx = vr*cos(Phi) - vphi*sin(Phi)
        vy.extend(vr * np.sin(Phi) + vphi * np.cos(Phi))  # vy = vr*sin(Phi) + vphi*cos(Phi)
        vz.extend(vz_data)  # vz remains the same

        # Collect weight and energy:
        weight.extend(data['weight'])
        energy.extend(data['energy'])

        # Check for optional keys 'denf4d' and 'denf4d_per_marker'
        if 'denf4d' in data:
            denf4d.extend(data['denf4d'])
        if 'denf4d_per_marker' in data:
            denf4d_per_marker.extend(data['denf4d_per_marker'])

    # Convert lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)
    weight = np.array(weight)
    energy = np.array(energy)

    # Handle optional arrays for denf4d and denf4d_per_marker
    denf4d = np.array(denf4d) if denf4d else np.zeros_like(energy)
    denf4d_per_marker = np.array(denf4d_per_marker) if denf4d_per_marker else np.zeros_like(energy)

    return X, Y, Z, vx, vy, vz, weight, energy, denf4d, denf4d_per_marker

def process_all_particles(birth_data, sink_data):
    """
    Processes both birth and sink particle data and combines results.

    Parameters:
        birth_data (list): List of dictionaries for birth particle data.
        sink_data (list): List of dictionaries for sink particle data.

    Returns:
        tuple: Arrays (X, Y, Z, vx, vy, vz, weights) with combined particle positions, velocities, and weights.
    """
    # Process birth data
    birth_X, birth_Y, birth_Z, birth_vx, birth_vy, birth_vz, birth_weight, birth_energy, birth_denf4d, birth_denf4d_per_marker = process_particle_data(birth_data)

    # Process sink data
    sink_X, sink_Y, sink_Z, sink_vx, sink_vy, sink_vz, sink_weight, sink_energy, sink_denf4d, sink_denf4d_per_marker = process_particle_data(sink_data)

    # Combine all data
    X = np.concatenate([birth_X, sink_X])
    Y = np.concatenate([birth_Y, sink_Y])
    Z = np.concatenate([birth_Z, sink_Z])
    vx = np.concatenate([birth_vx, sink_vx])
    vy = np.concatenate([birth_vy, sink_vy])
    vz = np.concatenate([birth_vz, sink_vz])
    weight = np.concatenate([birth_weight, sink_weight])
    energy = np.concatenate([birth_energy, sink_energy])
    denf4d = np.concatenate([birth_denf4d, sink_denf4d])
    denf4d_per_marker = np.concatenate([birth_denf4d_per_marker, sink_denf4d_per_marker])

    return X, Y, Z, vx, vy, vz, weight, energy, denf4d, denf4d_per_marker

def main():
    # Extract data:
    # ======================================================================================================================
    # birth_data and sink_data are lists of dictionaries.
    # Each element of the list corresponds to a specific HDF5 file.
    # There can be multiple birth and sink HDF5 files produced by FIDASIM when considering multistep CX events
    birth_data = process_source_files('birth', fida_run_dir)
    sink_data = process_source_files('sink', fida_run_dir)

    print("=================================================")
    print("Writing sources and sinks into text files...")
    print("=================================================")

    # Extract and combine all particle data:
    # ======================================================================================================================
    X, Y, Z, vx, vy, vz, weight, energy, denf4d, denf4d_per_marker = process_all_particles(birth_data, sink_data)

    # Get net flux, power and number of particles:
    net_flux = np.sum(weight)
    net_power = np.sum(weight * energy * consts.e)
    net_num_parts = weight.shape[0]

    # Get the NBI absorbed power:
    # ======================================================================================================================
    flux = birth_data[0]['flux']  # [ions/sec]
    nbi_abs_power = birth_data[0]['power']  # [W]

    # Get quantities resolved by 0th and 1st generation:
    # ======================================================================================================================
    n_birth_sets = len(birth_data)
    n_sink_sets = len(birth_data)
    n_birth = np.zeros(n_birth_sets)
    n_sink = np.zeros(n_sink_sets)
    birth_flux = np.zeros(n_birth_sets)
    sink_flux = np.zeros(n_sink_sets)
    birth_pwr = np.zeros(n_birth_sets)
    sink_pwr = np.zeros(n_sink_sets)

    for i in range(0, n_birth_sets, 1):
        try:
            n_birth[i] = int(birth_data[i]['weight'].shape[0])
            birth_flux[i] = birth_data[i]['flux']
            birth_pwr[i] = birth_data[i]['power']
        except:
            print(" ")
            print("birth_data[" + str(i) + "] doest not exist!")
            n_birth[i] = 0
            birth_flux[i] = 0
            birth_pwr[i] = 0
        try:
            n_sink[i] = int(sink_data[i]['weight'].shape[0])
            sink_flux[i] = sink_data[i]['flux']
            sink_pwr[i] = sink_data[i]['power']
        except:
            n_sink[i] = 0
            sink_flux[i] = 0
            sink_pwr[i] = 0
            print(" ")
            print("sink_data[" + str(i) + "] doest not exist!")

    # Get injected power:
    # ======================================================================================================================
    # Read cqlinput to get the NBI power injected:

    # Get cql_config:
    cql_run_dir = args.cql_run_dir.rstrip('/')
    cql_config = cql_run_dir + "/" + run_id + "_cql_config.nml"
    nml = f90nml.read(cql_config)

    # Read cqlinput:
    cqlinput = args.cql_run_dir + "/" + nml['cql3d_files']['cqlinput']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nml = f90nml.read(cqlinput)

    # Assign power:
    nbi_inj_power = nml['frsetup']['bptor'][0]  # [W]

    print(" ")
    print("==========================================================================================")
    print("Injected NBI power: " + str(nbi_inj_power * 1e-3) + " [kW]")

    # Print BIRTH data:
    print(" ")
    print(f"{'gen':<6} {'Birth Flux [p/s]':<20} {'Birth Power [kW]':<20} {'Birth N':<10}")
    print("-------------------------------------------------------")
    for i in range(len(n_birth)):
        # print(f"{i:<6} {birth_flux[i]:<20.6e} {birth_pwr[i]*1e-3:<20.6e} {int(n_birth[i]):<10}")
        print(f"{i:<6} {birth_flux[i]:<20.6e} {birth_pwr[i] * 1e-3:<20.3f} {int(n_birth[i]):<10}")

    # Print SINK data:
    print(" ")
    print(f"{'gen':<6} {'Sink Flux [p/s]':<20} {'Sink Power [kW]':<20} {'Sink N':<10}")
    print("-------------------------------------------------------")
    for i in range(len(n_sink)):
        print(f"{i:<6} {sink_flux[i]:<20.6e} {sink_pwr[i] * 1e-3:<20.3f} {int(n_sink[i]):<10}")

    print("==========================================================================================")

    print(" ")
    print("Net number of source points: " + str(net_num_parts) )

    print(" ")
    print("Injected NBI power: " + str(nbi_inj_power * 1e-6) + " [MW]")
    print("Absorbed NBI power: " + str(birth_pwr[0] * 1e-6) + " [MW]")
    print(f"Number of markers used: {int(n_birth[0])}")
    print(f"Total flux: {birth_flux[0]:.6e} [p/s]")
    print("=================================================")

    # Get the hostname and current date:
    # ======================================================================================================================
    hostname = socket.gethostname()
    current_date = datetime.now().strftime("%d-%b-%Y %H:%M:%S")  # Format: DD-MMM-YYYY HH:MM:SS

    # Saving file with birth points:
    # ======================================================================================================================
#     header =\
#     f"""host=sunfire08.pppl.gov                     date=09-Oct-2012 08:54:46
# "AC" FILE 128742N04.DATA9  TIME =  3.5229E-01 +/-  3.0000E-03 sec.
# tokamak: NSTX NLSYM=F #= 1 A= 2.0 Z= 1.0 NEUTRAL BEAM/DEPOSITION
# time data gathering from  3.537500E-01 to  3.557500E-01 sec.
# deposition function at guiding center.
# NEUTRAL BEAM #  = ALL AVAILABLE
# ENERGY FRACTION =  ALL
# POWER  INJECTED(this data sample) =  {nbi_inj_power:.6e} watts
# POWER DEPOSITED(this data sample) =  {birth_pwr[0]:.6e} watts         (0th GENERATION)
# POWER FOR ALL BEAMS =  {nbi_inj_power:.6e}  watts, not averaged over time
# beam toroidal angle XBZETA, deg.
# BEAM#  1-855.0000E-01
# BEAM#  2-820.0000E-01
# BEAM#  3-785.0000E-01
# beam nominal duct angle, deg.
# BEAM#  1 175.5000E+00
# BEAM#  2 172.0000E+00
# BEAM#  3 168.5000E+00
# N=  {net_num_parts:6}  #deposited/s= {np.sum(birth_flux):.6e}  #removed/s= {np.sum(sink_flux):.6e}  (NET quantities)
# N_shine_through =   12432
# Akima Hermite spline interpolation has been used
# 6(1x,1pe13.6)  get_fdep_beam
# x(cm)         y(cm)         z(cm)         v_x(cm/s)     v_y(cm/s)    v_z(cm/s)     weight(ions/s)
# <start-of-data>"""

    # Calculate padding to ensure "date=" starts at column 45
    padding_length = 44 - len(f"host={hostname}")
    padding = " " * max(padding_length, 0)

    # Fallback values if index 1 does not exist
    def safe_get(array, index, default=0.0):
        return array[index] if len(array) > index else default

    def safe_get_int(array, index, default=0):
        return int(array[index]) if len(array) > index else default

    # Example: assume all arrays are numpy arrays
    birth_flux_0 = safe_get(birth_flux, 0)
    birth_flux_1 = safe_get(birth_flux, 1)
    birth_pwr_0 = safe_get(birth_pwr, 0)
    birth_pwr_1 = safe_get(birth_pwr, 1)
    n_birth_0 = safe_get_int(n_birth, 0)
    n_birth_1 = safe_get_int(n_birth, 1)

    sink_flux_0 = safe_get(sink_flux, 0)
    sink_flux_1 = safe_get(sink_flux, 1)
    sink_pwr_0 = safe_get(sink_pwr, 0)
    sink_pwr_1 = safe_get(sink_pwr, 1)
    n_sink_0 = safe_get_int(n_sink, 0)
    n_sink_1 = safe_get_int(n_sink, 1)

#     header_part_1 =\
#     f"""host={hostname:<30}date={current_date:<20}
# {"GEN":<5} {"TYPE":<10} {"FLUX [p/s]":<20} {"PWR [W]":<20} {"num_parts":<10}
# {"-" * 80}
# {0:<5} {"BIRTH":<10} {birth_flux[0]:<20.6e} {birth_pwr[0]:<20.6e} {int(n_birth[0]):<10}
# {1:<5} {"BIRTH":<10} {birth_flux[1]:<20.6e} {birth_pwr[1]:<20.6e} {int(n_birth[1]):<10}
# {0:<5} {"SINK ":<10} {sink_flux[0]:<20.6e} {sink_pwr[0]:<20.6e} {int(n_sink[0]):<10}
# {1:<5} {"SINK ":<10} {sink_flux[1]:<20.6e} {sink_pwr[1]:<20.6e} {int(n_sink[1]):<10}
# """

    header_part_1 = \
    f"""host={hostname:<30}date={current_date:<20}
{"GEN":<5} {"TYPE":<10} {"FLUX [p/s]":<20} {"PWR [W]":<20} {"num_parts":<10}
{"-" * 80}
{0:<5} {"BIRTH":<10} {birth_flux_0:<20.6e} {birth_pwr_0:<20.6e} {int(n_birth_0):<10}
{1:<5} {"BIRTH":<10} {birth_flux_1:<20.6e} {birth_pwr_1:<20.6e} {int(n_birth_1):<10}
{0:<5} {"SINK ":<10} {sink_flux_0:<20.6e} {sink_pwr_0:<20.6e} {int(n_sink_0):<10}
{1:<5} {"SINK ":<10} {sink_flux_1:<20.6e} {sink_pwr_1:<20.6e} {int(n_sink_1):<10}
"""

    header_part_2 =\
    f"""POWER  INJECTED(this data sample) =  {nbi_inj_power:.6e} watts
POWER DEPOSITED(this data sample) =  {birth_pwr[0]:.6e} watts         (0th GENERATION)
POWER FOR ALL BEAMS =  {nbi_inj_power:.6e}  watts, not averaged over time
"""

    header_part_3 = "*\n" * 8

    header_part_4 = \
        f"""N=  {net_num_parts:6}  #deposited/s= {np.sum(birth_flux):.6e}  #removed/s= {np.sum(sink_flux):.6e}  (NET quantities)
N_shine_through =   12432 (Incorrect value)
*
*
x(cm)         y(cm)         z(cm)         v_x(cm/s)     v_y(cm/s)    v_z(cm/s)     weight(ions/s)     denf4d(ion/cm^3(s/cm)^3)     denf4d_per_marker(ion/cm^3(s/cm)^3)
<start-of-data>"""

    # Concatenate the parts to construct the full header:
    header = header_part_1 + header_part_2 + header_part_3 + header_part_4

    # Combine the arrays into a single 2D array:
    # ======================================================================================================================
    data = np.column_stack((X, Y, Z, vx, vy, vz, weight, denf4d, denf4d_per_marker))

    # Define the precision (number of decimal places)
    precision = 4#6  # Adjust this to your desired precision

    # Include scientific number formatting in the f-string
    fmt = f'% .{precision}e'

    # Specify the filename where you want to save the data
    # filename = cql_run_dir + "/" + "Ion_birth_points_FIDASIM.dat"
    filename = cql_run_dir + "/" + "ion_source_points_FIDASIM.dat"

    # Save the data to the file with the empty header and specified precision:
    # ======================================================================================================================
    print("Saving sources ...")
    np.savetxt(filename, data, delimiter=" ", fmt=fmt, header=header, comments="")
    print("Sources Saved!")
    time.sleep(5)

if __name__ == "__main__":
    main()
    print("End of script")
