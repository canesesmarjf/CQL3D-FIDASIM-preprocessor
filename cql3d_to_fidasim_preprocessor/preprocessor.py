import sys
import time
import os
import subprocess  # If subprocess calls are required in the script
import datetime
import warnings
import f90nml
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
import matplotlib

# Check if DISPLAY is available and set the appropriate backend
if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
    matplotlib.use('TkAgg')  # Use GUI-based backend
else:
    matplotlib.use('Agg')  # Use non-interactive backend for headless mode

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.ion()  # Turn on interactive mode (only affects GUI-based backends)
# Define physical constants:
mass_proton_CGI = 1.6726E-24 # CGI units [g]
charge_electron_SI = 1.60217663E-19  # SI units [C]
mass_proton_SI = 1.67262192E-27  # SI units [Kg]
mass_electron_SI = 9.1093837E-31  # SI units [Kg]

def set_fidasim_dir(path):
    """Set the FIDASIM directory and update the system path."""

    if not os.path.isdir(path):
        raise ValueError(f"The provided path {path} is not a valid directory.")

    global FIDASIM_dir
    FIDASIM_dir = path
    sys.path.append(os.path.join(FIDASIM_dir, 'lib/python'))

    try:
        global read_geqdsk, rz_grid, beam_grid, read_ncdf, fs
        from fidasim.utils import read_geqdsk, rz_grid, beam_grid, read_ncdf
        import fidasim as fs
    except ImportError as e:
        raise ImportError(f"Failed to import FIDASIM libraries: {e}")

def construct_interpolation_grid(config):
    print("     Running 'construct_interpolation_grid' ...")

    rmin = config["rmin"]
    rmax = config["rmax"]
    nr = config["nr"]
    zmin = config["zmin"]
    zmax = config["zmax"]
    nz = config["nz"]
    grid = rz_grid(rmin,rmax,nr,zmin,zmax,nz)
    return grid

def construct_fields(config, grid):
    print("     running 'construct_fields' ...")

    file_name = config["eqdsk_file_name"]
    eqdsk_type = config["eqdsk_type"]

    if eqdsk_type == 1:
        poloidal = True
    else:
        poloidal = False

    fields, rho, btipsign = read_geqdsk(file_name, grid, poloidal=poloidal)

    return fields, rho

def construct_plasma(config,grid,rho,plot_flag,plasma_from_cqlinput):
    print("     running 'construct_plasma' ...")

    if plasma_from_cqlinput:
        print("         Define profiles using cqlinput namelist ...")
        plasma = construct_plasma_from_cqlinput(config,grid,rho)
    else:
        print("         Define profiles using CQL3D standard netCDF output ...")
        plasma = construct_plasma_from_netcdf(config,grid,rho)

    return plasma

def construct_plasma_from_cqlinput(config,grid,rho):
    print("         reading cqlinput")

    # Read cqlinput namelist:
    cqlinput = config["cqlinput"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nml = f90nml.read(cqlinput)

    # Determine type of profile:
    iprone = nml['setup']['iprone']
    if iprone == "parabola":
        print("         Define plasma profile using cqlinput parabolic profile")
    elif iprone == "spline":
        print("         Define plasma profile using cqlinput spline profile")
        print("             Error: spline profile setup not available ...")
        sys.exit(0)
    else:
        print("         Error: Cannot define plasma ...")
        sys.exit(0)

    # Species information:
    ngen = nml['setup']['ngen']
    nmax = nml['setup']['nmax']
    fmass = nml['setup']['fmass']
    kspeci = nml['setup']['kspeci']
    bnumb = nml['setup']['bnumb']

    # Charge number of main impurity species:
    impurity_charge = config['impurity_charge']

    # Number of hydrogenic thermal species/isotopes
    nthermal = 1  # This appears to be set to 1 in the FIDASIM parameter list.

    # Mass of FP'd ion species from CQL3D:
    species_mass = np.array([fmass[0]/mass_proton_CGI])

    # rho clipped for rho>1 to avoid negative numbers:
    rho_clipped = np.clip(rho,0,1)

    #  Electron density:
    # ========================
    # Parabolic profile:
    dene_edge   = nml['setup']['reden'][1][1]
    dene_center = nml['setup']['reden'][0][1]
    npwr = nml['setup']['npwr']
    mpwr = nml['setup']['mpwr']
    dene = (dene_center - dene_edge)*(1.0 - rho_clipped**npwr[1])**mpwr[1] + dene_edge

    # Electron temperature:
    # ==========================
    te_edge = nml['setup']['temp'][1][1]
    te_center = nml['setup']['temp'][0][1]
    te = (te_center - te_edge) * (1.0 - rho_clipped ** npwr[1]) ** mpwr[1] + te_edge

    # Ion temperature:
    # =================
    ti_edge = nml['setup']['temp'][1][0]
    ti_center = nml['setup']['temp'][0][0]
    ti = (ti_center - ti_edge) * (1.0 - rho_clipped ** npwr[0]) ** mpwr[0] + ti_edge

    # Effective nuclear charge:
    # =========================
    zeff = np.ones(dene.shape)

    # Omega: plasma angular rotation:
    # ===============================
    omega_nc = np.zeros(dene.shape)

    # Values outside LCFS:
    # ====================
    # These are used to define non-zero plasma parameters in the region between the LCFS and the vacuum chamber wall
    # In the future, a better treatment of this can be done using exponential decay functions.
    dene_LCFS = config["dene_LCFS"]
    te_LCFS = config["te_LCFS"]
    ti_LCFS = config["ti_LCFS"]
    zeff_LCFS = config["zeff_LCFS"]
    rho_LCFS = config["rho_LCFS"]

    # Adjust profile values outside LCFS:
    # ====================================
    # Adjust the profile values in the regions beyond the CFS:
    dene = np.where(rho > rho_LCFS, dene_LCFS, dene)
    te = np.where(rho > rho_LCFS, te_LCFS, te)
    ti = np.where(rho > rho_LCFS, ti_LCFS, ti)
    zeff = np.where(rho > rho_LCFS, zeff_LCFS, zeff)

    # Ion and impurity density profiles:
    # Can "deni" be defined for multiple ion species?
    denimp = dene * (zeff - 1) / (impurity_charge * (impurity_charge - 1))
    deni = (dene - impurity_charge * denimp).reshape(1, grid['nr'], grid['nz'])
    deni = np.where(deni > 0.0, deni, 0.0).astype('float64')

    # Plasma rotation:
    # Set plasma rotation to zero
    # This may need to change in near future if ExB rotation becomes important
    vt = np.zeros(dene.shape)
    vr = vt
    vz = vt

    # Cold/Edge neutral density profile:
    # TODO: compute denn as in the case with netcdf file:
    denn = 1e8 * np.ones(dene.shape)  # Copied value used in the FIDASIM test case

    # Create mask:
    # FIDASIM requires a mask to define regions where the plasma is defined (mask == 1)
    # However, for tracking neutrals outside the LFCS and possibly all the way to the vacuum chamber wall
    # We need to make the plasma region defined over a large region.
    max_rho = 1.01
    max_rho = 3.5
    mask = np.where(rho <= max_rho, np.int64(1), np.int64(0))

    # Assemble output dictionary:
    plasma = {"time": 0.0, "data_source": config["cqlinput"], "mask": mask,
              "deni": deni, "denimp": denimp, "species_mass": species_mass,
              "nthermal": nthermal, "impurity_charge": impurity_charge,
              "te": te, "ti": ti, "vr": vr, "vt": vt, "vz": vz,
              "dene": dene, "zeff": zeff, "denn": denn}

    return plasma

    # Some variables that will be useful:

    # Profile switches:
    # iprote = nml['setup']['iprote']
    # iproti = nml['setup']['iproti']
    # iprone = nml['setup']['iprone']
    # iprozeff = nml['setup']['iprozeff']
    # iprovphi = nml['setup']['iprovphi']
    # ipronn = nml['setup']['ipronn']

    # Species information:
    # ngen = nml['setup']['ngen']
    # nmax = nml['setup']['nmax']
    # fmass = nml['setup']['fmass']
    # kspeci = nml['setup']['kspeci']
    # bnumb = nml['setup']['bnumb']

    # Parabolic profiles:
    # npwr = nml['setup']['npwr']
    # mpwr = nml['setup']['mpwr']
    # reden = nml['setup']['reden']
    # temp = nml['setup']['temp']

    # Spline profiles:
    # njene = nml['setup']['njene']
    # ryain = nml['setup']['ryain']
    # enein = nml['setup']['enein']
    # tein = nml['setup']['tein']
    # zeffin = nml['setup']['zeffin']
    # tiin = nml['setup']['tiin']

    # iprone.eq."parabola"): The user specifies reden(k,0) (central value) and reden(k,1) (edge value
    # nml['setup']['reden'][0] # Central value in [cm^-3]
    # nml['setup']['reden'][1] # edge value in [cm^-3]

def construct_plasma_from_netcdf(config,grid,rho):
    print("             reading netcdf file")

    # Read CQL3D standard .nc file into dict:
    nc_data = read_ncdf(config["plasma_file_name"])

    # Charge number of main impurity species:
    impurity_charge = config["impurity_charge"]

    # Number of hydrogenic thermal species/isotopes
    nthermal = 1  # This appears to be set to 1 in the FIDASIM parameter list.

    # Mass of FP ion species from CQL3D-M:
    species_mass = np.array([nc_data['fmass'][0] / mass_proton_CGI])

    # Grid coordinates for the nc profile data:
    r_nc = nc_data['solrz']  # R coordinates for flux surfaces
    z_nc = nc_data['solzz']  # Z coordinates for flux surfaces

    # Make z_nc symmetric about R-axis:
    flipped_z_nc = -np.flip(z_nc, axis=1)
    z_nc = np.concatenate([flipped_z_nc, z_nc[:, 1:]], axis=1)

    flipped_r_nc = np.flip(r_nc, axis=1)
    r_nc = np.concatenate([flipped_r_nc, r_nc[:, 1:]], axis=1)

    # ELECTRON DENSITY:
    # ==================
    # Find the time index of the latest valid dataset:
    threshold = 1e20
    valid_indices = np.where(nc_data['densz1'][:, :, :, 1] < threshold)[0]  # Check tdim values
    valid_idx = valid_indices[-1]  # Last valid index
    print("                 The last valid index is: " + str(valid_idx))

    # Electron density profile from nc file, dimensions [tdim, r0dim, zdim, species]
    dene_nc = nc_data['densz1'][valid_idx, :, :, 1]  # Selects the appropriate slice

    # Symmetrize it about the R axis:
    flipped_dene_nc = np.flip(dene_nc, axis=1)
    dene_nc = np.concatenate([flipped_dene_nc, dene_nc[:, 1:]], axis=1)

    # ELECTRON TEMPERATURE:
    # =====================
    # Electron temperature profile from nc file, dimensions [tdim, r0dim, species, zdim]
    te_nc = nc_data['energyz'][valid_idx, :, 1, :] * 2 / 3

    # Symmetrize it about the R axis:
    flipped_te_nc = np.flip(te_nc, axis=1)
    te_nc = np.concatenate([flipped_te_nc, te_nc[:, 1:]], axis=1)

    # ION TEMPERATURE:
    # ==================
    # Ion temperature profile from nc file, dimensions [tdim, r0dim, species, zdim]
    ti_nc = nc_data['energyz'][valid_idx, :, 0, :] * 2 / 3

    # Symmetrize it about the R axis:
    flipped_ti_nc = np.flip(ti_nc, axis=1)
    ti_nc = np.concatenate([flipped_ti_nc, ti_nc[:, 1:]], axis=1)

    # ZEFF: Effective Nuclear Charge:
    # ===============================
    # Set zeff to 1
    zeff_nc = np.ones(dene_nc.shape)

    # OMEGA: Plasma angular rotation:
    # ===============================
    # Set the angular rotation to 0
    omega_nc = np.zeros(dene_nc.shape)

    # Interpolation grid:
    r2d = grid['r2d']
    z2d = grid['z2d']

    # Values outside LCFS:
    # These are used to define non-zero plasma parameters in the region between the LCFS and the vacuum chamber wall
    # In the future, a better treatment of this can be done using exponential decay functions.
    dene_LCFS = config["dene_LCFS"]
    te_LCFS = config["te_LCFS"]
    ti_LCFS = config["ti_LCFS"]
    zeff_LCFS = config["zeff_LCFS"]
    rho_LCFS = config["rho_LCFS"]

    # Interpolate nc profiles into interpolation grid "grid":
    # Use "fill_value" option to prevent NaNs in regions outside the convex hull of the input data
    points = np.column_stack((r_nc.flatten(), z_nc.flatten()))  # Non-uniform grid points
    dene = griddata(points, dene_nc.flatten(), (r2d, z2d), method='linear', fill_value=dene_LCFS)
    zeff = griddata(points, zeff_nc.flatten(), (r2d, z2d), method='linear', fill_value=zeff_LCFS)
    te = griddata(points, te_nc.flatten(), (r2d, z2d), method='linear', fill_value=te_LCFS)
    ti = griddata(points, ti_nc.flatten(), (r2d, z2d), method='linear', fill_value=ti_LCFS)

    # Adjust the profile values in the regions beyond the CFS:
    # dene = np.where(rho > rho_LCFS, dene_LCFS, dene)
    # te = np.where(rho > rho_LCFS, te_LCFS, te)
    # ti = np.where(rho > rho_LCFS, ti_LCFS, ti)
    # zeff = np.where(rho > rho_LCFS, zeff_LCFS, zeff)

    # Add exponential decay to profiles:
    dene_min = dene_LCFS/100
    te_min = te_LCFS/100
    ti_min = ti_LCFS/100
    zeff_min = zeff_LCFS
    drho_profiles = 0.1
    dene = np.where(rho >= rho_LCFS,dene_min + (dene-dene_min) * 0.5 * (1 - np.tanh((rho-rho_LCFS)/drho_profiles)),dene)
    te   = np.where(rho >= rho_LCFS,te_min   +     (te-te_min) * 0.5 * (1 - np.tanh((rho-rho_LCFS)/drho_profiles)),te)
    ti   = np.where(rho >= rho_LCFS,ti_min   +     (ti-ti_min) * 0.5 * (1 - np.tanh((rho-rho_LCFS)/drho_profiles)),ti)
    zeff = np.where(rho >= rho_LCFS,zeff_min + (zeff-zeff_min) * 0.5 * (1 - np.tanh((rho-rho_LCFS)/drho_profiles)),zeff)

    # Ion and impurity density profiles:
    # Can "deni" be defined for multiple ion species?
    denimp = dene * (zeff - 1) / (impurity_charge * (impurity_charge - 1))
    deni = (dene - impurity_charge * denimp).reshape(1, grid['nr'], grid['nz'])
    deni = np.where(deni > 0.0, deni, 0.0).astype('float64')

    # Plasma rotation:
    # Set plasma rotation to zero
    # This may need to change in near future if ExB rotation becomes important
    vt = np.zeros(dene.shape)
    vr = vt
    vz = vt

    # Cold/Edge neutral density profile:
    denn_max = 1e10 # [cm^-3]
    denn_min = 1e6 # [cm^-3]
    drho_neutral = 0.2
    denn = denn_min + (denn_max-denn_min)*np.exp((rho-rho_LCFS)/drho_neutral)
    denn = np.where(rho<rho_LCFS,denn,denn_max).astype('float64')

    # Create mask:
    # FIDASIM requires a mask to define regions where the plasma is defined (mask == 1)
    # However, for tracking neutrals outside the LFCS and possibly all the way to the vacuum chamber wall
    # We need to make the plasma region defined over a large region.
    max_rho = 5.0
    mask = np.where(rho <= max_rho, np.int64(1), np.int64(0))

    # Assemble output dictionary:
    # plasma = {"time": nc_data['time'][-1], "data_source": config["plasma_file_name"], "mask": mask,
    #           "deni": deni, "denimp": denimp, "species_mass": species_mass,
    #           "nthermal": nthermal, "impurity_charge": impurity_charge,
    #           "te": te, "ti": ti, "vr": vr, "vt": vt, "vz": vz,
    #           "dene": dene, "zeff": zeff, "denn": denn, "profiles": nc_data}

    plasma = {"time": nc_data['time'][valid_idx], "data_source": config["plasma_file_name"], "mask": mask,
              "deni": deni, "denimp": denimp, "species_mass": species_mass,
              "nthermal": nthermal, "impurity_charge": impurity_charge,
              "te": te, "ti": ti, "vr": vr, "vt": vt, "vz": vz,
              "dene": dene, "zeff": zeff, "denn": denn, "time_index": valid_idx}

    return plasma

from scipy.interpolate import RegularGridInterpolator
def interpolate_fbm_over_velocity_space(fbm_nc, v_nc, t_nc, r_nc, z_nc, v_new, t_new):
    """
    Interpolates the 4D array `fbm_nc` on a new velocity (`v_new`) and theta (`t_new`) grid
    for each (R, Z) coordinate.

    Parameters:
    -----------
    fbm_nc : np.ndarray
        The input 4D array with dimensions [V, T, R, Z].
    v_nc : np.ndarray
        The original velocity grid (1D array).
    t_nc : np.ndarray
        The original theta grid (1D array).
    r_nc : np.ndarray
        The original radial coordinate grid (1D array).
    z_nc : np.ndarray
        The original axial coordinate grid (1D array).
    v_new : np.ndarray
        The new velocity grid (1D array) obtained from a uniform energy grid.
    t_new : np.ndarray
        The new theta grid (1D array) obtained from a uniform `cos(theta)` grid.

    Returns:
    --------
    fbm_interpolated : np.ndarray
        The interpolated 4D array with dimensions [len(v_new), len(t_new), len(r_nc), len(z_nc)].
    """
    # Define the shape of the output interpolated array
    v_len = len(v_new)
    t_len = len(t_new)
    r_len = len(r_nc)
    z_len = len(z_nc)

    # Initialize the output array
    fbm_interpolated = np.zeros((v_len, t_len, r_len, z_len))

    # Loop over each (R, Z) coordinate
    for r in range(r_len):
        for z in range(z_len):
            # Extract the 2D slice for the current (R, Z) location
            fbm_slice = fbm_nc[:, :, r, z]

            # Set up the interpolator for this slice
            interpolator = RegularGridInterpolator(
                (v_nc, t_nc), fbm_slice, method='linear',bounds_error=False, fill_value=None
            )

            # Create a meshgrid of the new velocity and theta points
            v_mesh, t_mesh = np.meshgrid(v_new, t_new, indexing='ij')
            points = np.array([v_mesh.flatten(), t_mesh.flatten()]).T

            # Perform the interpolation and reshape the result back to 2D
            interpolated_values = interpolator(points).reshape(v_len, t_len)

            # Store the interpolated values in the output array
            fbm_interpolated[:, :, r, z] = interpolated_values

    return fbm_interpolated

def construct_f4d(config,grid,rho,plot_flag,include_f4d):

    # NOTE:
    # The f4d provided by CQL3D has uniform grid in veloocity "v" and angle "t"
    # FIDASIM requires an f4d with uniform grid in energy "E" and pitch "p"
    # where E \propto v^2 and p \proto cos(t)
    # In order to correctly pass the distribution function to FIDASIM, we need to convert the CQL3D f4d into uniform E and p grids
    # This will require an 2D interpolation process

    print("     running 'construct_f4d' ...")
    start_time = time.time()

    # Define time stamp:
    # ====================================
    # Use time stamp from latest time in the netCDF file if it exists.
    # If it does not exist, then set it to 0:

    # Default time stamp:
    time_stamp = 0.0

    # Check if we have access to the required files:
    nc_file_name = config['plasma_file_name']
    f4d_ion_file_name = config['f4d_ion_file_name']
    f4d_electron_file_name = config['f4d_electron_file_name']
    nc_flag = os.path.exists(nc_file_name) and not os.path.isdir(nc_file_name)
    f4i_flag = os.path.exists(f4d_ion_file_name) and not os.path.isdir(f4d_ion_file_name)
    f4e_flag = os.path.exists(f4d_electron_file_name) and not os.path.isdir(f4d_electron_file_name)
    if include_f4d:
        if nc_flag and f4i_flag and f4e_flag:
            print("         All files for f4d interpolation are available!!")
        else:
            print("         Files for f4d interpolation missing ...")
            print("         Aborting python program at construct_f4d")
            sys.exit(1)
    else:
        print("         Skipping f4d interpolation ...")

    # Get time if nc file exists:
    if nc_flag:
        # Get time stamp and mass for this dataset:
        src_nc = read_ncdf(nc_file_name)
        threshold = 1e20
        valid_indices = np.where(src_nc['densz1'][:, :, :, 1] < threshold)[0]  # Check tdim values
        valid_idx = valid_indices[-1]  # Last valid index
        time_stamp = src_nc['time'][valid_idx]
    else:
        print("         Using default timestamp, plasma_file_name at " + nc_file_name)

    # Initialize fbm grid:
    # ==========================
    # FIDASIM subroutine "read_f" states that the fbm grid must have the same R,Z dimensions as the interpolation grid.
    # It can however, have any dimension for the energy and pitch angle dimensions.
    # To obtain the maximum resolution, we at least need to have the same energy and pitch dimensions as the nc file
    # We note that the pitch and energy grids for FIDASIM needs to be uniform.

    # Interpolation grids:
    r_grid = grid['r2d'] # [cm]
    z_grid = grid['z2d'] # [cm]

    # Initialize fbm array:
    num_R = r_grid.shape[0]
    num_Z = z_grid.shape[1]
    num_E = 2
    num_P = 2
    if include_f4d:
        num_E = 2*read_ncdf(config['f4d_ion_file_name'])['f4dv'].shape[0]
        num_P = read_ncdf(config['f4d_ion_file_name'])['f4dt'].shape[0]
    fbm_grid = np.zeros((num_R, num_Z, num_E, num_P))

    # Initialize fbm density:
    denf = np.zeros(z_grid.shape)

    if include_f4d:
        # Interpolate f4d into interpolation grids:
        # =========================================

        print("         Interpolating f4d ...")

        # Read nc file into dict:
        f4d_nc = read_ncdf(config['f4d_ion_file_name'])

        # Read distribution function from nc file, dimensions: [theta,v, Z, R]
        fbm_nc = f4d_nc['f4d']

        # Rearrange dimensions to be [v,theta,R,Z]:
        fbm_nc = fbm_nc.transpose(1, 0, 3, 2)

        # Velocity space 1D grids for fbm_nc:
        v_nc = f4d_nc['f4dv'] # Normalized to vnorm, [dimensionless]
        t_nc = f4d_nc['f4dt'] # [degrees]

        # Spatial 1D grids for fbm_nc:
        r_nc = f4d_nc['f4dr'] # [cm]
        z_nc = f4d_nc['f4dz'] # [cm]

        # Produce new v_nc and t_nc which corresponds to uniform E and P grids:

        # Get normalizing momentum per unit electron rest mass
        vnorm = f4d_nc['vnorm']  # [cm/s]
        vnorm_SI = vnorm/1e2  # [m/s]

        # Get corresponding normalizing  electron energy:
        enorm = f4d_nc['enorm'] # [keV]

        # Get minimum and maximum ion energy associated with input nc file:
        mass_AMU = np.array([src_nc['fmass'][0] / (mass_proton_SI * 1e3)])
        mass = mass_AMU[0] * mass_proton_SI  # [kg]
        v_max = v_nc.max()*vnorm_SI  # [m/s]
        e_max = 0.5*mass*(v_max**2)/charge_electron_SI # [eV]
        v_min = v_nc.min()*vnorm_SI  # [m/s]
        e_min = 0.5*mass*(v_min**2)/charge_electron_SI  # [eV]

        # Get the minimum and maximum pitch angle theta:
        t_max = t_nc.max()
        p_max = np.cos(t_max)
        t_min = t_nc.min()
        p_min = np.cos(t_min)

        # Create uniform E and P grids:
        de = (e_max - e_min)/(num_E-1)
        num_E = num_E-1
        e_1D = np.linspace(start=e_min+de/2,stop=e_max-de/2,num=num_E)*1e-3 # [keV]
        p_1D = np.linspace(start=p_min,stop=p_max,num=num_P) # [dimensionless = vpar/v]

        # Correct E grid:
        # Create E grid at center of cell so that the inverse transform method does not return negative energy values when sampling f4d
        # de = e_1D[1]-e_1D[0]
        # e_1D = e_1D + de/2

        # New (normalized) velocity and pitch angle grid:
        v_1D = (np.sqrt(2*charge_electron_SI*(e_1D*1e3)/mass)/vnorm_SI) # [dimensionless]
        t_1D = np.arccos(p_1D)

        # Interpolate fbm_nc at v_1D and t_1D
        fbm_interpolated = interpolate_fbm_over_velocity_space(fbm_nc, v_nc, t_nc, r_nc, z_nc, v_1D, t_1D)

        # Create velocity space mesh:
        vv, tt = np.meshgrid(v_1D, t_1D, indexing='ij')

        # Create mask: (for regions outside LCFS)
        max_rho = 1
        mask = np.where(rho <= max_rho, np.int64(1), np.int64(0))

        # Reshape fbm_nc to bring the R and Z dimensions to the front: [R,Z,E,P]
        fbm_reshaped = fbm_interpolated.transpose(2, 3, 0, 1)

        # Combine E and P coordinates into a single dimension: [R,Z,E*P]
        fbm_reshaped = fbm_reshaped.reshape(10, 121, -1)

        # Create the interpolator:
        input_points = (r_nc, z_nc)
        interpolator = RegularGridInterpolator(
            input_points,
            fbm_reshaped,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        # Create query points from interpolation grid, has dimensions [Rg*Zg,2]
        query_points = np.array([r_grid.flatten(),z_grid.flatten()]).T

        # Perform interpolation, has dimensions [Rg*Zg,E*P]: (This is the most time-consuming operation in this script)
        interpolated_values = interpolator(query_points)

        # Reshape interpolated values back into the grid shape: [Rg,Zg,E,P]
        # fbm_grid = interpolated_values.reshape(r_grid.shape[0], z_grid.shape[1], fbm_nc.shape[0], fbm_nc.shape[1])
        fbm_grid = interpolated_values.reshape(r_grid.shape[0], z_grid.shape[1], num_E, num_P)

        # Apply mask:
        fbm_grid *= mask[:,:,np.newaxis,np.newaxis]

    else:
        print("         skipping interpolating f4d ...")

    # Reorder dimensions back to original [E,P,Rg,Zg]
    fbm_grid = fbm_grid.transpose(2, 3, 0, 1)

    # Calculating fbm density:
    # =========================
    if include_f4d:
        print("         Integrating f4d ...")

        # Integrate over velocity (v) and pitch (v): (the trapz operation is quite time-consuming)
        p = p_1D # np.cos(t_1D)
        v = v_1D
        int_over_v = np.trapz(fbm_grid * v[:, np.newaxis, np.newaxis, np.newaxis] ** 2, x=v, axis=0)
        denf = -2 * np.pi * np.trapz(int_over_v, x=p, axis=0)

    else:
        print("         Skipping integrating f4d ...")

    # Assemble output dict:
    # =====================
    nenergy = fbm_grid.shape[0]
    npitch = fbm_grid.shape[1]

    # Are the following dimensions incorrectly set? JFCM 2024_11_12:
    pitch = np.zeros(nenergy)
    energy = np.zeros(npitch)

    if include_f4d:
        pitch = p_1D #pp_nc[0,:]
        energy = e_1D #ee_nc[:,1]

    fbm_dict = {"type":1,"time":time_stamp,"nenergy":nenergy,"energy":energy,"npitch":npitch,
              "pitch":pitch,"f":fbm_grid,"denf":denf,"data_source":os.path.abspath(config['f4d_ion_file_name'])}

    if plot_flag and include_f4d:
        # Plot #1: input f4d in vpar and vper coordinates:
        # ====================================================

        # Grids for vpar and vper coordinates:
        vv, tt = np.meshgrid(v_nc, t_nc, indexing='ij')
        vvpar = vv * np.cos(tt)
        vvper = vv * np.sin(tt)
        ee_nc = (0.5 * mass * (vv * vnorm_SI) ** 2) / (charge_electron_SI * 1e3)  # Energy grid in [keV]
        pp_nc = np.cos(tt)

        # Plot distribution function in vpar and vper coords:
        plt.figure()
        clight = 299792458 * 100  # [cm/s]
        vnc = vnorm / clight
        iz = np.int64(np.round(fbm_nc.shape[3]/2))
        ir = 0
        data = fbm_nc[:, :, ir, iz]

        log_data = np.full(data.shape, np.nan)  # Initialize log_data with NaNs
        log_data[data > 0] = np.log10(data[data > 0])
        #log_data = np.where(data > 0, np.log10(data), np.nan)
        contours = np.logspace(13, np.nanmax(log_data) * 1.1, 32)
        plt.contour(vvpar * vnc, vvper * vnc, data, colors='black', levels=contours)
        plt.contourf(vvpar * vnc, vvper * vnc, log_data, levels=np.log10(contours), cmap='viridis')
        plt.colorbar()
        plt.axis('image')
        plt.xlim([-0.01, 0.01])
        plt.ylim([0, 0.01])
        plt.title('Ion distribution function, R = 0, Z = 0')
        plt.savefig(config['output_path'] + 'f4d_ion_vpar_vper.png')

        # Plot #2: Compare input and interpolated f4d:
        # ============================================

        # Create a figure and two subplots, side by side (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Input f4d:
        ir = 0
        iz = np.int64(np.round(fbm_grid.shape[3]/2))
        data = fbm_grid[:, :, ir, iz]
        ee,pp = np.meshgrid(e_1D, p_1D, indexing='ij')
        ax1.contourf(ee, pp, data)
        ax1.set_xlim([0, enorm])
        ax1.set_xlabel('Energy [keV]')
        ax1.set_ylabel(r'pitch $v_{\parallel}/v$')
        ax1.set_title('INPUT: Ion f4d, R = 0, Z = 0 (to be used in FIDASIM)')

        # Output f4d:
        ir = 0
        iz = np.int64(np.round(fbm_nc.shape[3]/2))
        data = fbm_nc[:,:,ir,iz]
        ax2.contourf(ee_nc,pp_nc,data)
        ax2.set_xlim([0,enorm])
        ax2.set_xlabel('Energy [keV]')
        ax2.set_title('OUTPUT: Ion f4d, R = 0, Z = 0 (from CQL3D)')

        # Save figure:
        fig.savefig(config['output_path'] + 'f4d_ion_energy_pitch.png')

        # Plot #3: Compare ion density from integral:
        # ===========================================

        # Get density from source file:

        # Find the time index of the latest valid dataset:
        threshold = 1e20
        valid_indices = np.where(src_nc['densz1'][:, :, :, 1] < threshold)[0]  # Check tdim values
        valid_idx = valid_indices[-1]  # Last valid index

        r_src = src_nc['solrz']
        z_src = src_nc['solzz']
        dene_src = src_nc['densz1'][valid_idx,:,:,0]

        # Create figure object:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(wspace=0.5)  # Adjust the width space between subplots

        # Density from source file:
        cont1 = ax1.contourf(r_src,z_src,dene_src,vmin=0,vmax=1.6e13)
        ax1.set_xlim([0,15])
        ax1.set_ylim([0,200])
        fig.colorbar(cont1,ax=ax1)
        ax1.set_title('ne, from src file')

        # Density from f4d file:
        denf_clipped = np.clip(denf, 0, 1.6e13)
        cont2 = ax2.contourf(r_grid,z_grid,denf,vmin=0,vmax=1.6e13)
        ax2.set_xlim([0,15])
        ax2.set_ylim([0,200])
        fig.colorbar(cont2,ax=ax2)
        ax2.set_title('ne, from f4d file')

        # Save figure:
        fig.savefig(config['output_path'] + 'density_f4d_comparison.png')

        # Plot #4: Directly compare Z profiles:
        # ===================================

        # Create figure object:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(wspace=0.5)  # Adjust the width space between subplots

        ax1.plot(z_grid[0, :], denf[0, :],color='red',lw=4, label='f4d')
        ax1.plot(z_src[0,:],dene_src[0,:],color='black',lw=2, label='src')
        ax1.set_xlim([0,200])
        ax1.set_ylabel("density [cm^{-3}]")
        ax1.set_xlabel("Z [cm]")
        ax1.set_yscale('log')
        ax1.set_title('log scale: Z profile')
        ax1.legend()

        ax2.plot(z_grid[0, :], denf[0, :],color='red',lw=4, label='f4d')
        ax2.plot(z_src[0,:],dene_src[0,:],color='black',lw=2, label='src')
        ax2.set_xlim([0,200])
        ax2.set_ylabel("density [cm^{-3}]")
        ax2.set_xlabel("Z [cm]")
        ax2.set_yscale('linear')
        ax2.set_title('lin scale: Z profile')
        ax1.legend()

        # Save figure:
        fig.savefig(config['output_path'] + 'density_zprof_f4d_comparison.png')

        # Plot #5: Directly compare R profiles:
        # ===================================

        # Create figure object:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(wspace=0.5)  # Adjust the width space between subplots

        iz = np.int64(np.round(denf.shape[1]/2))
        ax1.plot(r_grid[:, iz], denf[:,iz], color='red', lw=4, label='f4d')
        ax1.plot(r_src[:, 0], dene_src[:,0], color='black', lw=2, label='src')
        ax1.set_xlim([0, 15])
        ax1.set_ylabel("density [cm^{-3}]")
        ax1.set_xlabel("Z [cm]")
        ax1.set_yscale('log')
        ax1.set_title('log scale: R profile')
        ax1.legend()

        ax2.plot(r_grid[:, iz], denf[:,iz], color='red', lw=4, label='f4d')
        ax2.plot(r_src[:, 0], dene_src[:,0], color='black', lw=2, label='src')
        ax2.set_xlim([0, 15])
        ax2.set_ylabel("density [cm^{-3}]")
        ax2.set_xlabel("Z [cm]")
        ax2.set_yscale('linear')
        ax2.set_title('lin scale: R profile')
        ax1.legend()

        # Save figure:
        fig.savefig(config['output_path'] + 'density_rprof_f4d_comparison.png')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"            Execution time: {execution_time} seconds")

    return fbm_dict

def construct_inputs(config, nbi):

    print("     running 'construct_inputs' ...")

    # Extract paths:
    cqlinput = config["cqlinput"]
    nc_file_name = config["plasma_file_name"]
    output_path = config["output_path"]

    # Get directory for FIDASIM installation
    fida_dir = fs.utils.get_fidasim_dir()

    # Read contents of cqlinput namelist file:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nml = f90nml.read(cqlinput)

    einj = nml['frsetup']['ebkev'][0] # [keV]
    pinj = nml['frsetup']['bptor'][0]*1e-6  # [MW]

    nprim = nml["frsetup"]["nprim"]
    fmass = nml["setup"]["fmass"][nprim-1]
    ab = fmass/(mass_proton_SI*1e3)

    current_fractions = np.zeros(3)
    for ii in range(3):
        current_fractions[ii] = nml['frsetup']['fbcur'][0][ii]

    if os.path.exists(nc_file_name) and not os.path.isdir(nc_file_name):
        # Get time stamp and mass for this dataset:
        src_nc = read_ncdf(nc_file_name)
        threshold = 1e20
        valid_indices = np.where(src_nc['densz1'][:, :, :, 1] < threshold)[0]  # Check tdim values
        valid_idx = valid_indices[-1]  # Last valid index
        time_stamp = src_nc['time'][valid_idx]
    else:
        time_stamp = 0.0

    # Set a seed to use for random number generator:
    seed = 1610735994

    # User defined inputs:
    calc_npa = 0
    calc_brems = 0
    calc_fida = 0
    calc_neutron = 0
    calc_cfpd = 0
    calc_res = 0
    calc_bes = 0
    calc_dcx = config["calc_dcx"]
    calc_halo = config["calc_halo"]
    calc_cold = 0
    calc_birth = config["calc_birth"]
    calc_fida_wght = 0
    calc_npa_wght = 0
    calc_pfida = 0
    calc_pnpa = 0

    # Number of particles to track:
    n_nbi = int(config["n_nbi"])
    n_halo = int(config["n_halo"])
    n_dcx = int(config["n_dcx"])
    n_birth = int(config["n_birth"])
    n_fida = int(1e3)
    n_npa = int(1e3)
    n_pfida = int(1e3)
    n_pnpa = int(1e3)

    basic_inputs = {"seed":seed, "device":"test", "shot":1, "time":time_stamp,
                    "einj":einj, "pinj":pinj, "current_fractions":current_fractions,
                    "ab":ab,
                    "lambdamin":647e0, "lambdamax":667e0, "nlambda":2000,
                    "n_fida":n_fida, "n_npa":n_npa, "n_nbi":n_nbi,
                    "n_pfida":n_pfida, "n_pnpa":n_pnpa,
                    "n_halo":n_halo, "n_dcx":n_dcx, "n_birth":n_birth,
                    "ne_wght":50, "np_wght":50,"nphi_wght":100,"emax_wght":100e0,
                    "nlambda_wght":1000,"lambdamin_wght":647e0,"lambdamax_wght":667e0,
                    "calc_npa":calc_npa, "calc_brems":calc_brems,"calc_fida":calc_fida,"calc_neutron":calc_neutron,
                    "calc_cfpd":calc_cfpd, "calc_res":calc_res,
                    "calc_bes":calc_bes, "calc_dcx":calc_dcx, "calc_halo":calc_halo, "calc_cold":calc_cold,
                    "calc_birth":calc_birth, "calc_fida_wght":calc_fida_wght,"calc_npa_wght":calc_npa_wght,
                    "calc_pfida":calc_pfida, "calc_pnpa":calc_pnpa,
                    "result_dir":output_path, "tables_file":fida_dir+'/tables/atomic_tables.h5'}

    # Define beam grid:
    basic_bgrid = {}
    nx = config["beam_grid"]["nx"]
    ny = config["beam_grid"]["ny"]
    nz = config["beam_grid"]["nz"]

    if (config["beam_grid"]["beam_aligned"]):
        # Beam-aligned beam grid:

        rstart = float(config["beam_grid"]["rstart"])
        length = float(config["beam_grid"]["length"])
        width = float(config["beam_grid"]["width"])
        height = float(config["beam_grid"]["height"])
        basic_bgrid = beam_grid(nbi, rstart=rstart, nx=nx, ny=ny, nz=nz, length=length, width=width, height=height)
    else:
        # Machine-aligned beam grid:

        # Origin of beam grid in UVW:
        basic_bgrid["origin"] = np.array(config['beam_grid']['origin_uvw'])

        # Bryan-Tait angles:
        # Rotation is applied to the UVW coord system to produce the beam grid coord system XYZ
        basic_bgrid["alpha"] = float(config["beam_grid"]["alpha"]) # Active rotation about "Z"
        basic_bgrid["beta"]  = float(config["beam_grid"]["beta"]) # Active rotation about "Y'"
        basic_bgrid["gamma"] = float(config["beam_grid"]["gamma"]) # Active rotation about "X''"

        # Define boundaries of beam grid in XYZ coordinate system:
        basic_bgrid["xmin"] = float(config["beam_grid"]["xmin"])
        basic_bgrid["xmax"] = float(config["beam_grid"]["xmax"])
        basic_bgrid["ymin"] = float(config["beam_grid"]["ymin"])
        basic_bgrid["ymax"] = float(config["beam_grid"]["ymax"])
        basic_bgrid["zmin"] = float(config["beam_grid"]["zmin"])
        basic_bgrid["zmax"] = float(config["beam_grid"]["zmax"])

        # Number of elements of beam grid:
        basic_bgrid["nx"] = nx
        basic_bgrid["ny"] = ny
        basic_bgrid["nz"] = nz

    # Add beam grid to input namelist:
    inputs = basic_inputs.copy()
    inputs.update(basic_bgrid)

    # Inputs related to non-thermal beam deposition, ion sources and sinks development:
    if "enable_nonthermal_calc" in config:
        inputs["enable_nonthermal_calc"] = config["enable_nonthermal_calc"]
    if "calc_sink" in config:
        inputs["calc_sink"] = config["calc_sink"]

    # Metadata on simulation run:
    inputs["comment"] = config["comment"]
    inputs["runid"] = config["runid"]

    return inputs

def construct_nbi(config):

    print("     running 'construct_nbi' ...")

    # Read contents of cqlinput namelist file:
    # Read in namelist:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nml = f90nml.read(config['cqlinput'])

    # Gather main variables from frsetup namelist:
    rpivot = nml['frsetup']['rpivot'][0]
    zpivot = nml['frsetup']['zpivot'][0]
    beta = nml['frsetup']['anglev'][0]*np.pi/180
    alpha = nml['frsetup']['angleh'][0]*np.pi/180
    blenp = nml['frsetup']['blenp'][0]
    bshape_nml = nml['frsetup']['bshape'][0]
    ashape_nml = nml['frsetup']['ashape'][0][0]
    naptr = nml['frsetup']['naptr']
    aheigh = nml['frsetup']['aheigh'][0][0]
    awidth = nml['frsetup']['awidth'][0][0]
    alen = nml['frsetup']['alen'][0][0]
    bheigh = nml['frsetup']['bheigh'][0]
    bwidth = nml['frsetup']['bwidth'][0]
    bhdiv = nml['frsetup']['bhdiv'][0] # Horizontal divergence in degrees
    bvdiv = nml['frsetup']['bvdiv'][0] # Vertical divergence in degrees
    bhfoc = nml['frsetup']['bhfoc'][0] # Horizontal focal length in cm
    bvfoc = nml['frsetup']['bvfoc'][0] # Vertical focal length in cm
    bhofset = nml['frsetup']['bhofset'][0] # Horizontal offset of aperture relative to beam centerline in cm
    bvofset = nml['frsetup']['bvofset'][0] # Vertical offset of aperture relative to beam centerline in cm


    # Calculate the rotation vector to convert XYZ (beam) coords to UVW (machine) coords:
    Arot = np.array([[np.cos(beta) , 0., np.sin(beta)],
                     [0.           , 1., 0.          ],
                     [-np.sin(beta), 0., np.cos(beta)]])

    Brot = np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                     [np.sin(alpha), np.cos(alpha) , 0.],
                     [0.           , 0.            , 1.]])

    Mrot = Brot@Arot

    # Standard coordinate system UVW (machine cartesian coordinates):
    s_hat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Beam-aligned coordinate system XYZ: [s_hat_pp] = [s_hat]*Mrot
    s_hat_pp = s_hat@Mrot

    # Displacement vector from pivot point to source location:
    xyz_d = [blenp, 0, 0]  # In beam-aligned coordinate system XYZ [s_hat_pp]
    uvw_d = Mrot @ xyz_d  # In machine coordinate system UVW [s_hat]

    # Pivot point vector in the machine coordinates UVW:
    uvw_pivot = np.array([rpivot, 0, zpivot])

    # Ion source position vector in UVW (machine) coords:
    uvw_src = uvw_pivot + uvw_d

    # Unit vector describing the direction of the beamline:
    x_hat = [1, 0, 0]
    x_hat_pp = Mrot @ x_hat
    uvw_axis = -x_hat_pp

    # Aperture:
    widy = 0.5*bwidth
    widz = 0.5*bheigh
    divy = np.array([1,1,1])*bhdiv*np.sqrt(1/2)*np.pi/180 # Radians
    divz = np.array([1,1,1])*bvdiv*np.sqrt(1/2)*np.pi/180 # Radians
    focy = bhfoc
    focz = bvfoc
    adist = alen
    awidy = 0.5*awidth
    awidz = 0.5*aheigh
    aoffy = bhofset
    aoffz = bvofset

    if naptr > 1:
        print("There are more than one apertures in cqlinput")
        # Might need to stop program if this is problematic
    naperture = naptr

    ashape = 1
    if ashape_nml == "s-rect":
        ashape = 1
    elif ashape_nml == "s-circ":
        ashape = 2

    bshape = 1
    if bshape_nml == "rect":
        bshape = 1
    elif bshape_nml == "circ":
        bshape = 2

    nbi = {"name":"test_beam","shape":bshape,"data_source":'run_tests:test_beam',
           "src":uvw_src, "axis":uvw_axis, "widy":widy, "widz":widz,
           "divy":divy, "divz":divz, "focy":focy, "focz":focz,
           "naperture":naperture, "ashape":ashape, "adist":adist,
           "awidy":awidy, "awidz":awidz, "aoffy":aoffy, "aoffz":aoffz}

    return nbi

def construct_fidasim_inputs_from_cql3d(config, plot_flag):

    print("running 'construct_fidasim_inputs_from_cql3d' ...")

    # Clear all figures:
    # Change to the desired directory
    # os.chdir(config["output_path"])
    # # Run the shell command to remove all .png files
    # subprocess.run("rm -f *.png", shell=True)

    # Define interpolation grid:
    # ==========================
    grid = construct_interpolation_grid(config)

    # Compute equil dict:
    # =====================
    equil, rho = construct_fields(config, grid)
    if plot_flag:
        plot_fields(config,grid,rho, equil)

    # Compute the plasma dict:
    # ========================
    plasma = construct_plasma(config,grid,rho,plot_flag,config['plasma_from_cqlinput'])
    if plot_flag:
        plot_plasma(config,grid,rho,plasma)

    # Compute fbm dict for ions:
    # =========================
    # How do I enable reading both ion and electron f?
    # What about for multiple ion species?
    fbm = construct_f4d(config,grid,rho,plot_flag,config['include_f4d'])

    # This may not be needed as this operation happens in read_f() subroutine in FIDASIM
    plasma['denf'] = fbm['denf']

    # Compute nbi dict from cqlinput:
    # ===============================
    nbi = construct_nbi(config)
    if plot_flag:
        fig,ax = plot_nbi(config,grid,rho,nbi)

    # Compute inputs dict:
    # ===================
    inputs = construct_inputs(config, nbi)
    if plot_flag:
        plot_beam_grid(fig,ax,config)

    # Produce input files for FIDASIM:
    # ================================
    print("     running 'prefida' ...")
    start_time = time.time()
    fs.prefida(inputs, grid, nbi, plasma, equil, fbm)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f" PREFIDA execution time: {execution_time} seconds")

    # Modify the *.dat output file produced by prefida and add new variables:
    # ======================================================================
    ps = os.path.sep
    input_file = inputs['result_dir'].rstrip(ps) + ps + inputs['runid'] + '_inputs.dat'
    write_fidasim_input_namelist(input_file,inputs)

def construct_preprocessor_config(fida_run_dir, cql_run_dir):

    # Check if fidasim run configuration file exists:
    fida_run_id = os.path.basename(os.path.normpath(fida_run_dir))
    fida_config_file = fida_run_dir.rstrip('/') + "/" + fida_run_id + "_config.nml"
    if not os.path.exists(fida_config_file):
        print(f"Error: The fidasim run configuration file '{fida_config_file}' does not exist.")
        sys.exit(1)

    # Check if cql run configuration file exists:
    cql_run_id = os.path.basename(os.path.normpath(cql_run_dir))
    cql_config_file = cql_run_dir.rstrip('/') + "/" + cql_run_id + "_cql_config.nml"
    if not os.path.exists(cql_config_file):
        print(f"Error: The cql3d run configuration file '{cql_config_file}' does not exist.")
        sys.exit(1)

    # Read contents of the run configuration namelist files:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fida_nml = f90nml.read(fida_config_file)
        cql_nml  = f90nml.read(cql_config_file)

    # Initialize user input dictionary:
    config = {}

    # Metadata for fidasim run:
    config["runid"] = fida_run_id
    config["comment"] = fida_nml['fidasim_run_info']['comment']

    # R-Z Interpolation grid:
    sub_nml = fida_nml['rz_interpolation_grid']
    config["rmin"] = sub_nml['rmin']
    config["rmax"] = sub_nml['rmax']
    config["nr"] = sub_nml['nr']
    config["zmin"] = sub_nml['zmin']
    config["zmax"] = sub_nml['zmax']
    config["nz"] = sub_nml['nz']

    # CQL3DM related files:
    # (Assume we only have a single ion species)
    # sub_nml = fida_nml['cql3d_input_files']
    # config['input_dir'] = sub_nml['input_dir']
    # input_dir =  config['input_dir']
    # config["cqlinput"] = input_dir + sub_nml['cqlinput']

    config['input_dir'] = cql_run_dir
    config["cqlinput"] = cql_run_dir + cql_nml['cql3d_files']['cqlinput']
    config["eqdsk_file_name"] = cql_run_dir + cql_nml['cql3d_files']['eqdsk_file_name']
    config["eqdsk_type"] = cql_nml['cql3d_files']['eqdsk_type']

    # Define "plasma_from_cqlinput" flag:
    if "plasma_from_cqlinput" in fida_nml['cql3d_input_files']:
        config["plasma_from_cqlinput"] = fida_nml['cql3d_input_files']['plasma_from_cqlinput']
    else:
        config["plasma_from_cqlinput"] = False

    # Get plasma file name:
    if config["plasma_from_cqlinput"] == False:
        config["plasma_file_name"] = cql_run_dir + cql_nml['cql3d_files']['plasma_file_name']
    else:
        config["plasma_file_name"] = ''

    # Define "include_f4d" flag:
    if "include_f4d" in fida_nml['cql3d_input_files']:
        config["include_f4d"] = fida_nml['cql3d_input_files']['include_f4d']
    else:
        config["include_f4d"] = False

    if config['include_f4d'] == True:
        config["f4d_ion_file_name"] = cql_run_dir + cql_nml['cql3d_files']['f4d_ion_file_name']
        config["f4d_electron_file_name"] = cql_run_dir + cql_nml['cql3d_files']['f4d_electron_file_name']
    else:
        config["f4d_ion_file_name"] = ''
        config["f4d_electron_file_name"] = ''

    # Plasma profiles parameters:
    sub_nml = fida_nml['plasma_profiles_params']
    config["impurity_charge"] = sub_nml['impurity_charge']
    config["rho_LCFS"] = sub_nml['rho_LCFS']
    config["dene_LCFS"] = sub_nml['dene_LCFS']
    config["te_LCFS"] = sub_nml['te_LCFS']
    config["ti_LCFS"] = sub_nml['ti_LCFS']
    config["zeff_LCFS"] = sub_nml['zeff_LCFS']

    # Define path to output directory:
    # (This is where you want the FIDASIM HDF5 files to be written to)
    config['output_path'] = fida_nml['preprocessor_output']['output_dir']
    if not config['output_path']:
        config['output_path'] = fida_run_dir

    # Beam physics switches:
    sub_nml = fida_nml['beam_physics_switches']
    if "enable_nonthermal_calc" in sub_nml:
        config['enable_nonthermal_calc'] = sub_nml['enable_nonthermal_calc']
    if "calc_sink" in sub_nml:
        config['calc_sink'] = sub_nml['calc_sink']

    config['calc_birth'] = sub_nml['calc_birth']
    config['calc_dcx'] = sub_nml['calc_dcx']
    config['calc_halo'] = sub_nml['calc_halo']

    # Define number of Monte-Carlo particles:
    sub_nml = fida_nml['monte_carlo_particles']
    config['n_nbi'] = sub_nml['n_nbi']
    config['n_birth'] = sub_nml['n_birth']
    config['n_dcx'] = sub_nml['n_dcx']
    config['n_halo'] = sub_nml['n_halo']

    # Define beam grid parameters:
    sub_nml = fida_nml['beam_grid']
    config["beam_grid"] = {}
    config["beam_grid"]["beam_aligned"] = sub_nml['beam_aligned']  # Option between machine-aligned or beam-aligned
    config["beam_grid"]["nx"] = sub_nml['nx']
    config["beam_grid"]["ny"] = sub_nml['ny']
    config["beam_grid"]["nz"] = sub_nml['nz']

    # Only used when "beam_aligned" == True
    if config["beam_grid"]["beam_aligned"]:
        config["beam_grid"]["rstart"] = sub_nml['rstart']
        config["beam_grid"]["length"] = sub_nml['length']
        config["beam_grid"]["width"] = sub_nml['width']
        config["beam_grid"]["height"] = sub_nml['height']
    else:
        # Only used when "beam_aligned" == False
        config["beam_grid"]["alpha"] = sub_nml['alpha']
        config["beam_grid"]["beta"] = sub_nml['beta']
        config["beam_grid"]["gamma"] = sub_nml['gamma']
        config["beam_grid"]["xmin"] = sub_nml['xmin']
        config["beam_grid"]["xmax"] = sub_nml['xmax']
        config["beam_grid"]["ymin"] = sub_nml['ymin']
        config["beam_grid"]["ymax"] = sub_nml['ymax']
        config["beam_grid"]["zmin"] = sub_nml['zmin']
        config["beam_grid"]["zmax"] = sub_nml['zmax']
        config["beam_grid"]["origin_uvw"] = sub_nml['origin_uvw']

    return config

def write_fidasim_input_namelist(filename, inputs):
    """
    # JFCM 2024_08_15:
    # This function is based on FIDASIM's write_namelist function in PREFIDA
    # It has been modified to produce the namelist required for our purposes
    #
    #+#write_namelist
    #+Writes namelist file
    #+***
    #+##Input Arguments
    #+     **filename**: Name of the namelist file
    #+
    #+     **inputs**: Input dictionary
    #+
    #+##Example Usage
    #+```python
    #+>>> write_namelist(filename, inputs)
    #+```
    """
    fs.utils.info("Writing namelist file...")

    fidasim_version = fs.utils.get_version(fs.utils.get_fidasim_dir())

    with open(filename, "w") as f:
        f.write("!! Created: {}\n".format(datetime.datetime.now()))
        f.write("!! FIDASIM version: {}\n".format(fidasim_version))
        f.write("!! Comment: {}\n".format(inputs['comment']))
        f.write("&fidasim_inputs\n\n")

        f.write("!! Shot Info\n")
        f.write("shot = {:d}    !! Shot Number\n".format(inputs['shot']))
        f.write("time = {:f}    !! Time [s]\n".format(inputs['time']))
        f.write("runid = '{}'   !! runID\n".format(inputs['runid']))
        f.write("result_dir = '{}'    !! Result Directory\n\n".format(inputs['result_dir']))

        f.write("!! Input Files\n")
        f.write("tables_file = '{}'   !! Atomic Tables File\n".format(inputs['tables_file']))
        f.write("equilibrium_file = '" + inputs['equilibrium_file'] + "'    !! File containing plasma parameters and fields\n")
        f.write("geometry_file = '" + inputs['geometry_file'] + "'    !! File containing NBI and diagnostic geometry\n")
        f.write("distribution_file = '" + inputs['distribution_file'] + "'    !! File containing fast-ion distribution\n\n")

        f.write("!! Simulation Switches\n")
        f.write("calc_bes = {:d}    !! Calculate NBI Spectra\n".format(inputs['calc_bes']))
        f.write("calc_dcx = {:d}    !! Calculate Direct CX Spectra\n".format(inputs['calc_dcx']))
        f.write("calc_halo = {:d}    !! Calculate Halo Spectra\n".format(inputs['calc_halo']))
        f.write("calc_cold = {:d}    !! Calculate Cold D-alpha Spectra\n".format(inputs['calc_cold']))
        f.write("calc_brems = {:d}    !! Calculate Bremsstrahlung\n".format(inputs['calc_brems']))
        f.write("calc_fida = {:d}    !! Calculate FIDA Spectra\n".format(inputs['calc_fida']))
        f.write("calc_npa = {:d}   !! Calculate NPA\n".format(inputs['calc_npa']))
        f.write("calc_pfida = {:d}    !! Calculate Passive FIDA Spectra\n".format(inputs['calc_pfida']))
        f.write("calc_pnpa = {:d}   !! Calculate Passive NPA\n".format(inputs['calc_pnpa']))
        f.write("calc_neutron = {:d}   !! Calculate B-T Neutron Rate\n".format(inputs['calc_neutron']))
        f.write("calc_cfpd = {:d}   !! Calculate B-T CFPD Energy Resolved Count Rate\n".format(inputs['calc_cfpd']))
        f.write("calc_birth = {:d}    !! Calculate Birth Profile\n".format(inputs['calc_birth']))
        f.write("calc_fida_wght = {:d}    !! Calculate FIDA weights\n".format(inputs['calc_fida_wght']))
        f.write("calc_npa_wght = {:d}    !! Calculate NPA weights\n".format(inputs['calc_npa_wght']))
        f.write("calc_res = {:d}    !! Calculate spatial resolution\n".format(inputs['calc_res']))

        if "enable_nonthermal_calc" in inputs:
            # Add new inputs when running non-thermal calculations:
            f.write("\n!! Non-thermal beam deposition switches\n")
            f.write("enable_nonthermal_calc = {:d} !! Enable the use of f4d to calculate beam deposition\n".format(inputs['enable_nonthermal_calc']))
        if "calc_sink" in inputs:
            f.write("calc_sink = {:d} !! Calculate ion sink process\n".format(inputs['calc_sink']))

        f.write("\n!! Advanced Settings\n")
        f.write("seed = {:d}    !! RNG Seed. If seed is negative a random seed is used\n".format(inputs['seed']))
        f.write("flr = {:d}    !! Turn on Finite Larmor Radius corrections\n".format(inputs['flr']))
        f.write("load_neutrals = {:d}    !! Load neutrals from neutrals file\n".format(inputs['load_neutrals']))
        f.write("output_neutral_reservoir = {:d}    !! Output neutral reservoir to neutrals file\n".format(inputs['output_neutral_reservoir']))
        f.write("neutrals_file = '" + inputs['neutrals_file'] + "'    !! File containing the neutral density\n")
        f.write("stark_components = {:d}    !! Output stark components\n".format(inputs['stark_components']))
        f.write("verbose = {:d}    !! Verbose\n\n".format(inputs['verbose']))

        f.write("!! Monte Carlo Settings\n")
        f.write("n_fida = {:d}    !! Number of FIDA mc particles\n".format(inputs['n_fida']))
        f.write("n_npa = {:d}    !! Number of NPA mc particles\n".format(inputs['n_npa']))
        f.write("n_pfida = {:d}    !! Number of Passive FIDA mc particles\n".format(inputs['n_pfida']))
        f.write("n_pnpa = {:d}    !! Number of Passive NPA mc particles\n".format(inputs['n_pnpa']))
        f.write("n_nbi = {:d}    !! Number of NBI mc particles\n".format(inputs['n_nbi']))
        f.write("n_halo = {:d}    !! Number of HALO mc particles\n".format(inputs['n_halo']))
        f.write("n_dcx = {:d}     !! Number of DCX mc particles\n".format(inputs['n_dcx']))
        f.write("n_birth = {:d}    !! Number of BIRTH mc particles\n\n".format(inputs['n_birth']))

        f.write("!! Neutral Beam Settings\n")
        f.write("ab = {:f}     !! Beam Species mass [amu]\n".format(inputs['ab']))
        f.write("pinj = {:f}     !! Beam Power [MW]\n".format(inputs['pinj']))
        f.write("einj = {:f}     !! Beam Energy [keV]\n".format(inputs['einj']))
        f.write("current_fractions(1) = {:f} !! Current Fractions (Full component)\n".format(inputs['current_fractions'][0]))
        f.write("current_fractions(2) = {:f} !! Current Fractions (Half component)\n".format(inputs['current_fractions'][1]))
        f.write("current_fractions(3) = {:f} !! Current Fractions (Third component)\n\n".format(inputs['current_fractions'][2]))

        f.write("!! Beam Grid Settings\n")
        f.write("nx = {:d}    !! Number of cells in X direction (Into Plasma)\n".format(inputs['nx']))
        f.write("ny = {:d}    !! Number of cells in Y direction\n".format(inputs['ny']))
        f.write("nz = {:d}    !! Number of cells in Z direction\n".format(inputs['nz']))
        f.write("xmin = {:f}     !! Minimum X value [cm]\n".format(inputs['xmin']))
        f.write("xmax = {:f}     !! Maximum X value [cm]\n".format(inputs['xmax']))
        f.write("ymin = {:f}     !! Minimum Y value [cm]\n".format(inputs['ymin']))
        f.write("ymax = {:f}     !! Maximum Y value [cm]\n".format(inputs['ymax']))
        f.write("zmin = {:f}     !! Minimum Z value [cm]\n".format(inputs['zmin']))
        f.write("zmax = {:f}     !! Maximum Z value [cm]\n\n".format(inputs['zmax']))

        f.write("!! Tait-Bryan Angles for z-y`-x`` rotation\n")
        f.write("alpha = {:f}     !! Rotation about z-axis [rad]\n".format(inputs['alpha']))
        f.write("beta  = {:f}     !! Rotation about y`-axis [rad]\n".format(inputs['beta']))
        f.write("gamma = {:f}     !! Rotation about x``-axis [rad]\n\n".format(inputs['gamma']))

        f.write("!! Beam Grid origin in machine coordinates (cartesian)\n")
        f.write("origin(1) = {:f}     !! U value [cm]\n".format(inputs['origin'][0]))
        f.write("origin(2) = {:f}     !! V value [cm]\n".format(inputs['origin'][1]))
        f.write("origin(3) = {:f}     !! W value [cm]\n\n".format(inputs['origin'][2]))

        f.write("!! Wavelength Grid Settings\n")
        f.write("nlambda = {:d}    !! Number of Wavelengths\n".format(inputs['nlambda']))
        f.write("lambdamin = {:f}    !! Minimum Wavelength [nm]\n".format(inputs['lambdamin']))
        f.write("lambdamax = {:f}    !! Maximum Wavelength [nm]\n\n".format(inputs['lambdamax']))

        f.write("!! Weight Function Settings\n")
        f.write("ne_wght = {:d}    !! Number of Energies for Weights\n".format(inputs['ne_wght']))
        f.write("np_wght = {:d}    !! Number of Pitches for Weights\n".format(inputs['np_wght']))
        f.write("nphi_wght = {:d}    !! Number of Gyro-angles for Weights\n".format(inputs['nphi_wght']))
        f.write("emax_wght = {:f}    !! Maximum Energy for Weights [keV]\n".format(inputs['emax_wght']))
        f.write("nlambda_wght = {:d}    !! Number of Wavelengths for Weights \n".format(inputs['nlambda_wght']))
        f.write("lambdamin_wght = {:f}    !! Minimum Wavelength for Weights [nm]\n".format(inputs['lambdamin_wght']))
        f.write("lambdamax_wght = {:f}    !! Maximum Wavelength for Weights [nm]\n\n".format(inputs['lambdamax_wght']))

        f.write("!! Adaptive Time Step Settings\n")
        f.write("adaptive = {:d}    !! Adaptive switch, 0:split off, 1:dene, 2:denn, 3:denf, 4:deni, 5:denimp, 6:te, 7:ti\n".format(inputs['adaptive']))
        f.write("split_tol = {:f}    !! Tolerance for change in plasma parameter, number of cell splits is proportional to 1/split_tol\n".format(inputs['split_tol']))
        f.write("max_cell_splits = {:d}    !! Maximum number of times a cell can be split\n\n".format(inputs['max_cell_splits']))
        f.write("/\n\n")

    fs.utils.success("Namelist file created: {}\n".format(filename))

def plot_plasma(config,grid,rho,plasma):
    print("         Plotting plasma profiles:")
    # Plot mask:
    fig = plt.figure(5)
    plt.contourf(grid['r2d'], grid['z2d'], plasma['mask'])
    plt.colorbar()
    levels = np.linspace(0.01, 1, 10)
    contour_plot = plt.contour(grid['r2d'], grid['z2d'], rho, levels=levels, colors='red', linewidths=1.0)
    plt.clabel(contour_plot, inline=True, fontsize=8)
    # plt.xlim([0,25])
    plt.xlabel('R [cm]')
    plt.title('Mask')
    fig.set_size_inches(4, 6)  # Width, Height in inches
    plt.savefig(config["output_path"] + 'mask.png')

    # Plot density profile:
    fig = plt.figure(6)
    plt.contourf(grid['r2d'], grid['z2d'], plasma['dene'] * plasma['mask'])
    plt.colorbar()
    levels = np.linspace(0.01, 1, 10)
    contour_plot = plt.contour(grid['r2d'], grid['z2d'], rho, levels=levels, colors='red', linewidths=1.0)
    plt.clabel(contour_plot, inline=True, fontsize=8)
    # plt.xlim([0,25])
    plt.xlabel('R [cm]')
    plt.title('Electron density')
    fig.set_size_inches(4, 6)  # Width, Height in inches
    plt.savefig(config["output_path"] + 'dene.png')

    # Plot electron temperature profile:
    fig = plt.figure(7)
    plt.contourf(grid['r2d'], grid['z2d'], plasma['te'] * plasma['mask'])
    plt.colorbar()
    levels = np.linspace(0.01, 1, 10)
    contour_plot = plt.contour(grid['r2d'], grid['z2d'], rho, levels=levels, colors='red', linewidths=1.0)
    plt.clabel(contour_plot, inline=True, fontsize=8)
    # plt.xlim([0, 25])
    plt.xlabel('R [cm]')
    plt.title('Electron temperature [keV]')
    fig.set_size_inches(4, 6)  # Width, Height in inches
    plt.savefig(config["output_path"] + 'te.png')

    # Plot ion temperature profile:
    fig = plt.figure(8)
    plt.contourf(grid['r2d'], grid['z2d'], plasma['ti'] * plasma['mask'])
    plt.colorbar()
    levels = np.linspace(0.01, 1, 10)
    contour_plot = plt.contour(grid['r2d'], grid['z2d'], rho, levels=levels, colors='red', linewidths=1.0)
    plt.clabel(contour_plot, inline=True, fontsize=8)
    # plt.xlim([0, 25])
    plt.xlabel('R [cm]')
    plt.title('Ion temperature [keV]')
    fig.set_size_inches(4, 6)  # Width, Height in inches
    plt.savefig(config["output_path"] + 'ti.png')

    # Plot neutral density profile:
    fig = plt.figure(9)
    plt.contourf(grid['r2d'], grid['z2d'], plasma['denn'] * plasma['mask'])
    plt.colorbar()
    levels = np.linspace(0.01, 1, 10)
    contour_plot = plt.contour(grid['r2d'], grid['z2d'], rho, levels=levels, colors='red', linewidths=1.0)
    plt.clabel(contour_plot, inline=True, fontsize=8)
    # plt.xlim([0,25])
    plt.xlabel('R [cm]')
    plt.title('Neutral density')
    fig.set_size_inches(4, 6)  # Width, Height in inches
    plt.savefig(config["output_path"] + 'denn.png')

    # Plot product of neutral and electron density profile:
    fig = plt.figure(10)
    plt.contourf(grid['r2d'], grid['z2d'], plasma['denn'] * plasma['dene'] * plasma['mask'])
    plt.colorbar()
    levels = np.linspace(0.01, 1, 10)
    contour_plot = plt.contour(grid['r2d'], grid['z2d'], rho, levels=levels, colors='red', linewidths=1.0)
    plt.clabel(contour_plot, inline=True, fontsize=8)
    # plt.xlim([0,25])
    plt.xlabel('R [cm]')
    plt.title('Product of neutral and electron density')
    fig.set_size_inches(4, 6)  # Width, Height in inches
    plt.savefig(config["output_path"] + 'denn_x_dene.png')

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

def draw_rectangle(ax, center, axis, widy, widz):
    '''Draws a rectangle in 3D space given the center, axis, and half-widths.'''
    # Generate two orthogonal vectors to the axis
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    y_vec = np.cross(axis, [0, 0, 1])
    if np.linalg.norm(y_vec) == 0:
        y_vec = np.cross(axis, [0, 1, 0])
    y_vec = y_vec / np.linalg.norm(y_vec) * widy

    z_vec = np.cross(axis, y_vec)
    z_vec = z_vec / np.linalg.norm(z_vec) * widz

    # Calculate the four corners of the rectangle
    corners = [
        center - y_vec - z_vec,
        center + y_vec - z_vec,
        center + y_vec + z_vec,
        center - y_vec + z_vec
    ]

    # Add rectangle to plot
    verts = [corners]
    poly = Poly3DCollection(verts, facecolors='b', alpha=0.6, edgecolors='r')
    ax.add_collection3d(poly)

def draw_circle(ax, center, axis, widy, widz):
    '''Draws a circle in 3D space given the center, axis, and radii (widy, widz).'''
    # Generate two orthogonal vectors to the axis
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    y_vec = np.cross(axis, [0, 0, 1])
    if np.linalg.norm(y_vec) == 0:
        y_vec = np.cross(axis, [0, 1, 0])
    y_vec = y_vec / np.linalg.norm(y_vec) * widy

    z_vec = np.cross(axis, y_vec)
    z_vec = z_vec / np.linalg.norm(z_vec) * widz

    # Parametric angles for the circle
    u = np.linspace(0, 2 * np.pi, 100)

    # Circle points in the normal plane
    circle_points = np.outer(np.cos(u), y_vec) + np.outer(np.sin(u), z_vec)

    # Translate the circle to the center point
    circle_points = circle_points + center

    # Plot the circle
    ax.plot3D(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='b')

def draw_nbi_aperture(ax, shape, center, axis, widy, widz):
    '''Draw a neutral beam ion source (NBI) aperture as either a rectangle (shape=1) or a circle (shape=2).'''

    # Check the shape type
    if shape == 1:
        # Draw a rectangular source/aperture:
        draw_rectangle(ax, center=center, axis=axis, widy=widy, widz=widz)
    elif shape == 2:
        # Draw a circular source/aperture:
        draw_circle(ax, center=center, axis=axis, widy=widy, widz=widz)
    else:
        raise ValueError("Shape must be 1 (rectangular) or 2 (circular).")

def plot_nbi(config,grid,rho,nbi):
    print("         Plot NBI geometry")

    # NBI optical path:
    # =================
    src_uvw = nbi['src']
    axis_uvw = nbi['axis']
    path_distance = 2.5*np.max([grid['r'].max(), grid['z'].max()])
    path = np.linspace(0,path_distance)
    nbi_path_uvw = src_uvw + np.outer(path,axis_uvw)
    aper_uvw = nbi['adist']

    # Plasma boundary:
    # =================
    dum, ax = plt.subplots()
    contour = ax.contour(grid['r2d'],grid['z2d'], rho, levels=[1], colors='r')
    contour_paths = contour.collections[0].get_paths()  # Get the contour paths
    Rc = []
    Zc = []
    # Loop over the paths and extract the coordinates
    for path in contour_paths:
        v = path.vertices  # Array of coordinates
        Rc.append(v[:, 0])  # R coordinates
        Zc.append(v[:, 1])  # Z coordinates

    # Convert Rc and Zc to 1D arrays (assuming one continuous contour)
    Rc = np.concatenate(Rc)
    Zc = np.concatenate(Zc)
    plt.close(dum)

    # Create a surface of revolution about the W axis:
    # ================================================================
    # Number of points around the Z-axis
    num_angles = 100
    theta_start = -0*np.pi/180
    theta_end = +360*np.pi/180
    theta = np.linspace(theta_start, theta_end, num_angles)

    # Create a 2D grid for theta and Rc
    Theta, Rc_grid = np.meshgrid(theta, Rc)

    # Parametric equations for the surface of revolution
    U = Rc_grid * np.cos(Theta)  # X = R cos(theta)
    V = Rc_grid * np.sin(Theta)  # Y = R sin(theta)
    W = np.tile(Zc, (num_angles, 1)).T  # Z stays the same for each revolution

    # Plot 3D diagram:
    # =================
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    u_path = nbi_path_uvw[:,0]
    v_path = nbi_path_uvw[:,1]
    w_path = nbi_path_uvw[:,2]
    ax.plot(u_path,v_path,w_path,color='r',lw=2)

    # Plot surface with transparency (alpha)
    ax.plot_surface(U, V, W, color='b', alpha=0.25)

    # Set labels
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_zlabel('W')

    # Set equal scaling on all axes
    set_axes_equal(ax)

    # Extract variables from the dictionary
    shape = nbi['shape']
    src = nbi['src']  # The source location in UVW coordinates
    axis = nbi['axis']  # The normal vector of the ion source
    widy = nbi['widy']  # Half-width of the source in y direction
    widz = nbi['widz']  # Half-width of the source in z direction

    # Plot the source geometry
    draw_nbi_aperture(ax, shape, src, axis, widy, widz)

    # Draw aperture:
    asrc = src + axis*nbi['adist']
    draw_nbi_aperture(ax, nbi['ashape'], asrc, axis, nbi['awidy'], nbi['awidz'])

    # Set the view:
    elev = +30
    azim = -60
    ax.view_init(elev, azim)
    plt.show()

    # Format axes:
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis.pane.set_alpha(0.3)
    ax.grid(False)
    plt.show()

    # Save image:
    ax.set_title('NBI geometry')
    fig.set_size_inches(9, 9)  # Width, Height in inches
    fig.savefig(config["output_path"] + 'NBI_geometry_isometric.png')

    # Set the view:
    elev = +90
    azim = -90
    ax.view_init(elev, azim)
    plt.show()

    # Save image:
    fig.savefig(config["output_path"] + 'NBI_geometry_UV.png')

    # Set the view:
    elev = +0
    azim = -90
    ax.view_init(elev, azim)
    plt.show()

    # Save image:
    fig.savefig(config["output_path"] + 'NBI_geometry_UW.png')

    # Set the view:
    elev = +0
    azim = +0
    ax.view_init(elev, azim)
    plt.show()

    # Save image:
    fig.savefig(config["output_path"] + 'NBI_geometry_VW.png')

    return fig,ax

def draw_rectangular_volume(ax, vertices_uvw, faces, alpha=0.3, color='b'):
    '''Draws a rectangular volume in 3D using the vertices and faces.'''

    # Create a list of the vertices for each face
    verts = []
    for face in faces.T:
        verts.append([vertices_uvw[:, i] for i in face])

    # Create a Poly3DCollection with translucent faces
    poly = Poly3DCollection(verts, facecolors=color, linewidths=0.5, edgecolors=color, alpha=alpha)

    # Add the collection to the plot
    ax.add_collection3d(poly)

def plot_beam_grid(fig,ax,config):
    print("         plotting beam grid:")

    # Gather grid boundaries in XYZ coordinate system:
    xmin = config['beam_grid']['xmin']
    xmax = config['beam_grid']['xmax']
    ymin = config['beam_grid']['ymin']
    ymax = config['beam_grid']['ymax']
    zmin = config['beam_grid']['zmin']
    zmax = config['beam_grid']['zmax']

    # Bryan-Tait rotation angles to produce XYZ from UVW:
    alpha = config['beam_grid']['alpha']
    beta = config['beam_grid']['beta']
    gamma = config['beam_grid']['gamma']

    # Beam grid origin in machine coordinate system UVW:
    beam_grid_origin_uvw = np.array([0,0,0])

    # Vertices in XYZ coord system:
    beam_grid_vertices = np.array([
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax]
    ])

    # Transpose to get 8 column vectors
    beam_grid_vertices = beam_grid_vertices.T  # Transpose to get column vectors

    # Define the beam_grid faces of the rectangle using the beam_grid vertices
    beam_grid_faces = np.array([
        [1, 2, 4, 3],  # Side face (xmin)
        [5, 6, 8, 7],  # Side face (xmax)
        [1, 2, 6, 5],  # Side face (ymin)
        [3, 4, 8, 7],  # Side face (ymax)
        [1, 3, 7, 5],  # Bottom face
        [2, 4, 8, 6]  # Top face
    ])

    # In Python, indices are zero-based, so subtract 1 from MATLAB-style indexing:
    beam_grid_faces = beam_grid_faces.T - 1

    # Compute cosines and sines for the rotation matrix
    ca = np.cos(alpha)
    cb = np.cos(beta)
    sa = np.sin(alpha)
    sb = np.sin(beta)

    # Rotation matrix (basis) as described in the MATLAB code
    beam_grid_basis = np.array([
        [ca * cb, -sa, ca * sb],
        [cb * sa, ca, sa * sb],
        [-sb, 0, cb]
    ])

    # Compute the beam grid vertices in the UVW machine coordinate system
    beam_grid_vertices_uvw = np.zeros_like(beam_grid_vertices)
    for ii in range(beam_grid_vertices.shape[1]):
        beam_grid_vertices_uvw[:, ii] = beam_grid_origin_uvw + np.dot(beam_grid_basis, beam_grid_vertices[:, ii])

    # Draw the rectangular volume with translucent surfaces
    draw_rectangular_volume(ax, beam_grid_vertices_uvw, beam_grid_faces, alpha=0.3, color='cyan')

    elev = +30
    azim = -60
    ax.view_init(elev, azim)
    plt.show()

    # Save image:
    ax.set_title('NBI + beam grid geometry')
    fig.set_size_inches(9, 9)  # Width, Height in inches
    fig.savefig(config["output_path"] + 'NBI_geometry_isometric.png')

    # Set the view:
    elev = +90
    azim = -90
    ax.view_init(elev, azim)
    plt.show()

    # Save image:
    fig.savefig(config["output_path"] + 'NBI_geometry_UV.png')

    # Set the view:
    elev = +0
    azim = -90
    ax.view_init(elev, azim)
    plt.show()

    # Save image:
    fig.savefig(config["output_path"] + 'NBI_geometry_UW.png')

    # Set the view:
    elev = +0
    azim = +0
    ax.view_init(elev, azim)
    plt.show()

    # Save image:
    fig.savefig(config["output_path"] + 'NBI_geometry_VW.png')

def plot_fields(config,grid,rho, equil):
    print("         Plotting field related data:")

    output_path = config["output_path"]

    fig = plt.figure(1)
    plt.contourf(grid["r2d"], grid["z2d"], equil["bz"])
    plt.colorbar
    plt.title("Bz_2D")
    plt.ylabel("Z [cm]")
    plt.xlabel("R [cm]")
    fig.set_size_inches(4, 6)  # Width, Height in inches
    plt.savefig(output_path + 'Bz_2D.png')
    # plt.show()

    fig = plt.figure(2)
    plt.contourf(grid["r2d"], grid["z2d"], equil["br"])
    plt.colorbar
    plt.title("Br_2D")
    plt.ylabel("Z [cm]")
    plt.xlabel("R [cm]")
    fig.set_size_inches(4, 6)  # Width, Height in inches
    plt.savefig(output_path + 'Br_2D.png')
    # plt.show()

    fig = plt.figure(3)
    levels = np.linspace(0.01, 1, 10)
    contour_plot = plt.contour(grid['r2d'], grid['z2d'], rho, levels=levels, colors='red', linewidths=1.0)
    plt.clabel(contour_plot, inline=True, fontsize=8)

    # plt.contour(grid["r2d"],grid["z2d"],rho,levels=np.linspace(0.01, 1.0, 16))
    # plt.colorbar
    plt.title("flux_surfaces")
    plt.ylabel("Z [cm]")
    plt.xlabel("R [cm]")
    fig.set_size_inches(4, 6)  # Width, Height in inches
    plt.savefig(output_path + 'flux_surfaces.png')
    # plt.show()

    # fig = plt.figure(4)
    # plt.contourf(grid['r2d'], grid['z2d'], plasma['dene'].T * plasma['mask'].T)
    # plt.contour(grid['r2d'], grid['z2d'], rho, levels=[1.0, 1.2, 1.3, 1.4])
    # fig.set_size_inches(4, 6)  # Width, Height in inches

def main():
    print("Add some code")

if __name__=='__main__':
    main()