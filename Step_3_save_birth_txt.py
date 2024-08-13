# In this script, we read the birth.h5 file produced and create a .dat text file that can be read back into CQL3DM
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import scipy.constants as consts
import h5py

# birth point file path:
data_dir = "./Step_1_output/"
file_path_births = data_dir + "WHAM_example_birth.h5"

def print_hdf5_structure(name, obj):
    print(name)
    for key, value in obj.attrs.items():
        print(f"  Attribute {key}: {value}")

with h5py.File(file_path_births, 'r') as file:
    # file.visititems(print_hdf5_structure)
    # Particle data:
    n_birth = file['n_birth'][()]
    energy = file['energy'][:]*1e3 # [eV]
    weight = file['weight'][:] # [ions/s]
    ri = file['ri'][:] # [cm]
    vi = file['vi'][:] # [cm/s]

# Extract particle positions:
R = ri[:,0]
Z = ri[:,1]
Phi = ri[:,2]

# Convert positions to cartesian coordinates:
X = R*np.cos(Phi)
Y = R*np.sin(Phi)

# Extract particle velocities:
vr = vi[:,0]
vz = vi[:,1]
vphi = vi[:,2]
v_perp = np.sqrt( vr**2 + vphi**2)

# Convert positions to cartesian coordinates:
vx = vr*np.cos(Phi) - vphi*np.sin(Phi)
vy = vr*np.sin(Phi) + vphi*np.cos(Phi)

# Calculate absorbed power:
flux = np.sum(weight) # [ions/sec]
nbi_abs_power = np.sum(energy*weight*consts.e) # [W]
nbi_inj_power = 100e3 # [W]

print("Injected NBI power: " + str(nbi_inj_power*1e-6) + " [MW]" )
print("Absorbed NBI power: " + str(nbi_abs_power*1e-6) + " [MW]" )
print(f"Number of markers used: {n_birth}")
print(f"Total flux: {flux:.6e} [p/s]")

# Saving file with birth points:
# ------------------------------
header =\
f"""host=sunfire08.pppl.gov                     date=09-Oct-2012 08:54:46
"AC" FILE 128742N04.DATA9  TIME =  3.5229E-01 +/-  3.0000E-03 sec.
tokamak: NSTX NLSYM=F #= 1 A= 2.0 Z= 1.0 NEUTRAL BEAM/DEPOSITION
time data gathering from  3.537500E-01 to  3.557500E-01 sec.
deposition function at guiding center.
NEUTRAL BEAM #  = ALL AVAILABLE
ENERGY FRACTION =  ALL
POWER  INJECTED(this data sample) =  {nbi_inj_power:.6e} watts
POWER DEPOSITED(this data sample) =  {nbi_abs_power:.6e} watts
POWER FOR ALL BEAMS =  {nbi_inj_power:.6e}  watts, not averaged over time
beam toroidal angle XBZETA, deg.
BEAM#  1-855.0000E-01
BEAM#  2-820.0000E-01
BEAM#  3-785.0000E-01
beam nominal duct angle, deg.
BEAM#  1 175.5000E+00
BEAM#  2 172.0000E+00
BEAM#  3 168.5000E+00
N=  {n_birth:6}  #deposited/s= {flux:.6e}   rng_seed=     641880641
N_shine_through =   12432
Akima Hermite spline interpolation has been used
6(1x,1pe13.6)  get_fdep_beam
x(cm)         y(cm)         z(cm)         v_x(cm/s)     v_y(cm/s)    v_z(cm/s)     weight(ions/s)
<start-of-data>"""

# Combine the arrays into a single 2D array
# data = np.column_stack((birth["pos"][:,0], birth["pos"][:,1], birth["pos"][:,2], birth["v"][:,0], birth["v"][:,1], birth["v"][:,2]))
data = np.column_stack((X,Y,Z,vx,vy,vz,weight))

# Define the precision (number of decimal places)
precision = 6  # Adjust this to your desired precision

# Consider E format as well:

# Include scientific number formatting in the f-string
fmt = f'% .{precision}e'

# Specify the filename where you want to save the data
filename = data_dir + "birth_points_FIDASIM_f1.dat"

# Combine the empty header lines with the data
# header = "\n".join(empty_header_lines)

# Save the data to the file with the empty header and specified precision
np.savetxt(filename, data, delimiter=" ", fmt=fmt, header=header, comments="")


print("End of script")
