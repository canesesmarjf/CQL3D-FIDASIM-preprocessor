import sys
import time
import os
import datetime
import warnings
import f90nml
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode

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

def read_fields(file_name,grid):

    print("     running 'read_fields' ...")
    fields, rho, btipsign = read_geqdsk(file_name, grid, poloidal=True)
    return fields, rho

def read_plasma(config,grid,rho,plot_flag=False):

    print("     running 'read_plasma' ...")

    # Read CQL3D standard .nc file into dict:
    nc_data = read_ncdf(config["plasma_file_name"])

    # Charge number of main impurity species:
    impurity_charge = config["impurity_charge"]

    # Number of hydrogenic thermal species/isotopes
    nthermal = 1 # This appears to be set to 1 in the FIDASIM parameter list.

    # Mass of FP ion species from CQL3D-M:
    species_mass = np.array([nc_data['fmass'][0]/mass_proton_CGI])

    # Grid coordinates for the nc profile data:
    r_nc = nc_data['solrz'] # R coordinates for flux surfaces
    z_nc = nc_data['solzz'] # Z coordinates for flux surfaces

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
    print("The last valid index is: " + str(valid_idx))

    # Electron density profile from nc file, dimensions [tdim, r0dim, zdim, species]
    dene_nc = nc_data['densz1'][valid_idx, :, :, 1]  # Selects the appropriate slice
    # dene_nc = nc_data['densz1'][-1,:,:,1]

    # Symmetrize it about the R axis:
    flipped_dene_nc = np.flip(dene_nc, axis=1)
    dene_nc = np.concatenate([flipped_dene_nc, dene_nc[:, 1:]], axis=1)

    # ELECTRON TEMPERATURE:
    # =====================
    # Electron temperature profile from nc file, dimensions [tdim, r0dim, species, zdim]
    te_nc = nc_data['energyz'][valid_idx,:,1,:]*2/3
    # te_nc = nc_data['energyz'][-1,:,1,:]*2/3

    # Symmetrize it about the R axis:
    flipped_te_nc = np.flip(te_nc, axis=1)
    te_nc = np.concatenate([flipped_te_nc, te_nc[:, 1:]], axis=1)

    # ION TEMPERATURE:
    # ==================
    # Ion temperature profile from nc file, dimensions [tdim, r0dim, species, zdim]
    ti_nc = nc_data['energyz'][valid_idx,:,0,:]*2/3
    # ti_nc = nc_data['energyz'][-1,:,0,:]*2/3

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
    dene = griddata(points, dene_nc.flatten(), (r2d, z2d), method='linear',fill_value=dene_LCFS)
    zeff = griddata(points, zeff_nc.flatten(), (r2d, z2d), method='linear',fill_value=zeff_LCFS)
    te = griddata(points, te_nc.flatten(), (r2d, z2d), method='linear',fill_value=te_LCFS)
    ti = griddata(points, ti_nc.flatten(), (r2d, z2d), method='linear',fill_value=ti_LCFS)

    # Adjust the profile values in the regions beyond the CFS:
    dene = np.where(rho > rho_LCFS, dene_LCFS, dene)
    te = np.where(rho > rho_LCFS, te_LCFS, te)
    ti = np.where(rho > rho_LCFS, ti_LCFS, ti)
    zeff = np.where(rho > rho_LCFS, zeff_LCFS, zeff)

    # Ion and impurity density profiles:
    # Can "deni" be defined for multiple ion species?
    denimp = dene*(zeff - 1)/(impurity_charge*(impurity_charge-1))
    deni = (dene - impurity_charge*denimp).reshape(1,grid['nr'],grid['nz'])
    deni = np.where(deni > 0.0, deni, 0.0).astype('float64')

    # Plasma rotation:
    # Set plasma rotation to zero
    # This may need to change in near future if ExB rotation becomes important
    vt = np.zeros(dene.shape)
    vr = vt
    vz = vt

    # Cold/Edge neutral density profile:
    denn = 1e8*np.ones(dene.shape) # Copied value used in the FIDASIM test case

    # Create mask:
    # FIDASIM requires a mask to define regions where the plasma is defined (mask == 1)
    # However, for tracking neutrals outside the LFCS and possibly all the way to the vacuum chamber wall
    # We need to make the plasma region defined over a large region.
    max_rho = 10
    mask = np.where(rho <= max_rho, np.int64(1), np.int64(0))

    # Assemble output dictionary:
    plasma = {"time":nc_data['time'][-1], "data_source":config["plasma_file_name"], "mask":mask,
                "deni":deni,"denimp":denimp,"species_mass":species_mass,
                "nthermal":nthermal,"impurity_charge":impurity_charge,
                "te":te, "ti":ti, "vr":vr, "vt":vt, "vz":vz,
                "dene":dene, "zeff":zeff, "denn":denn,"profiles":nc_data}

    if plot_flag:
        fig = plt.figure(5)
        plt.contourf(r2d,z2d,dene*mask)
        plt.colorbar()
        plt.plot(r_nc.T,z_nc.T,color='r',lw=1.0)
        plt.xlim([0,25])
        plt.xlabel('R [cm]')
        plt.title('Electron density')
        fig.set_size_inches(4, 6)  # Width, Height in inches
        plt.savefig(config["output_path"] + 'dene.png')

        fig = plt.figure(6)
        plt.contourf(r2d, z2d, te * mask)
        plt.colorbar()
        plt.plot(r_nc.T, z_nc.T, color='r', lw=1.0)
        plt.xlim([0, 25])
        plt.xlabel('R [cm]')
        plt.title('Electron temperature [keV]')
        fig.set_size_inches(4, 6)  # Width, Height in inches
        plt.savefig(config["output_path"] + 'te.png')

        fig = plt.figure(7)
        plt.contourf(r2d,z2d,ti*mask)
        plt.colorbar()
        plt.plot(r_nc.T,z_nc.T,color='r',lw=1.0)
        plt.xlim([0,25])
        plt.xlabel('R [cm]')
        plt.title('Ion temperature [keV]')
        fig.set_size_inches(4, 6)  # Width, Height in inches
        plt.savefig(config["output_path"] + 'ti.png')

    return plasma

def read_f4d(config,grid,rho,plot_flag=False,interpolate_f4d=True):

    # Physical constants in SI units:
    # charge_electron_SI = 1.60217663E-19  # [C]
    # mass_proton_SI = 1.67262192E-27  # [Kg]
    # mass_electron_SI = 9.1093837E-31  # [Kg]

    print("     running 'read_f4d' ...")
    start_time = time.time()

    # Read nc file into dict:
    f4d_nc = read_ncdf(config['f4d_ion_file_name'])

    # Get distribution function, dimensions: [theta,v, Z, R]
    fbm_nc = f4d_nc['f4d']

    # Rearrange dimensions to be [v,theta,R,Z]:
    fbm_nc = fbm_nc.transpose(1, 0, 3, 2)

    # Velocity space 1D grids for fbm_nc:
    v_nc = f4d_nc['f4dv']
    t_nc = f4d_nc['f4dt']
    vv, tt = np.meshgrid(v_nc, t_nc, indexing='ij')

    # Spatial 1D grids for fbm_nc:
    r_nc = f4d_nc['f4dr']
    z_nc = f4d_nc['f4dz']

    # Create mask: (for regions outside LCFS)
    # =======================================
    max_rho = 1
    mask = np.where(rho <= max_rho, np.int64(1), np.int64(0))

    # Interpolate f4d into interpolation grids:
    # =========================================

    # Interpolation grids:
    r_grid = grid['r2d']
    z_grid = grid['z2d']

    # Initialize fbm array:
    num_R = r_grid.shape[0]
    num_Z = z_grid.shape[1]
    num_E = fbm_nc.shape[0]
    num_P = fbm_nc.shape[1]
    fbm_grid = np.zeros((num_R, num_Z, num_E, num_P))

    if interpolate_f4d == True:
        print("         Interpolating f4d ...")

        # Reshape fbm_nc to bring the R and Z dimensions to the front: [R,Z,E,P]
        fbm_reshaped = fbm_nc.transpose(2, 3, 0, 1)

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

        # Perform interpolation, has dimensions [Rg*Zg,E*P]
        interpolated_values = interpolator(query_points)

        # Reshape interpolated values back into the grid shape: [Rg,Zg,E,P]
        fbm_grid = interpolated_values.reshape(r_grid.shape[0], z_grid.shape[1], fbm_nc.shape[0], fbm_nc.shape[1])
    else:
        print("         skipping interpolating f4d ...")

    # Reorder dimensions back to original [E,P,Rg,Zg]
    fbm_grid = fbm_grid.transpose(2, 3, 0, 1)

    # Apply mask to fbm_grid:
    mask_reshaped = mask[np.newaxis, np.newaxis, :, :]
    fbm_grid *= mask_reshaped

    # Calculating fbm density:
    # =========================

    # Initialize fbm density:
    denf = np.zeros(z_grid.shape)

    if (interpolate_f4d == True):
        print("         Integrating f4d ...")
        # Integrate over velocity (v) and pitch (v):
        p = np.cos(t_nc)
        v = v_nc
        int_over_v = np.trapz(fbm_grid * v[:, np.newaxis, np.newaxis, np.newaxis] ** 2, x=v, axis=0)
        denf = -2 * np.pi * np.trapz(int_over_v, x=p, axis=0)
    else:
        print("         Skipping integrating f4d ...")

    # Get time stamp and mass for this dataset:
    src_nc = read_ncdf(config['plasma_file_name'])
    time_stamp = src_nc['time'][-1]
    mass_AMU = np.array([src_nc['fmass'][0]/(mass_proton_SI*1e3)])

    # Grids for E and pitch coordinates (This is what FIDASIM requires)
    vnorm = f4d_nc['vnorm']
    mass = mass_AMU*mass_proton_SI
    enorm = f4d_nc['enorm']
    ee_nc = (0.5 * mass * (vv * vnorm * 1e-2) ** 2) / (charge_electron_SI * 1e3)  # Energy grid in [keV]
    pp_nc = np.cos(tt)

    # Assemble output dict:
    # =====================
    nenergy = fbm_grid.shape[0]
    npitch = fbm_grid.shape[1]
    pitch = pp_nc[0,:]
    energy = ee_nc[:,1]

    fbm_dict = {"type":1,"time":time_stamp,"nenergy":nenergy,"energy":energy,"npitch":npitch,
              "pitch":pitch,"f":fbm_grid,"denf":denf,"data_source":os.path.abspath(config['f4d_ion_file_name'])}

    if plot_flag:
        # Plot #1: input f4d in vpar and vper coordinates:
        # ====================================================

        # Grids for vpar and vper coordinates:
        vvpar = vv * np.cos(tt)
        vvper = vv * np.sin(tt)

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
        ax1.contourf(ee_nc, pp_nc, data)
        ax1.set_xlim([0, enorm])
        ax1.set_xlabel('Energy [keV]')
        ax1.set_ylabel(r'pitch $v_{\parallel}/v$')
        ax1.set_title('INPUT: Ion f4d, R = 0, Z = 0')

        # Output f4d:
        ir = 0
        iz = np.int64(np.round(fbm_nc.shape[3]/2))
        data = fbm_nc[:,:,ir,iz]
        ax2.contourf(ee_nc,pp_nc,data)
        ax2.set_xlim([0,enorm])
        ax2.set_xlabel('Energy [keV]')
        ax2.set_title('OUTPUT: Ion f4d, R = 0, Z = 0')

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

def assemble_inputs(config, nbi):

    print("     running 'assemble_inputs' ...")

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

    # Read standard cql3d output nc file:
    nc = read_ncdf(nc_file_name)
    time_stamp = nc['time'][-1]

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
    rstart = float(config["beam_grid"]["rstart"])
    length = float(config["beam_grid"]["length"])
    width = float(config["beam_grid"]["width"])
    height = float(config["beam_grid"]["height"])

    if (config["beam_grid"]["beam_aligned"]):
        # Beam-aligned beam grid:
        basic_bgrid = beam_grid(nbi, rstart=rstart, nx=nx, ny=ny, nz=nz, length=length, width=width, height=height)
    else:
        # Machine-aligned beam grid:
        basic_bgrid["nx"] = nx
        basic_bgrid["ny"] = ny
        basic_bgrid["nz"] = nz
        basic_bgrid["alpha"] = np.pi
        basic_bgrid["beta"] = np.pi/2
        basic_bgrid["gamma"] = 0.0
        basic_bgrid["xmin"] = -length/2
        basic_bgrid["xmax"] = +length/2
        basic_bgrid["ymin"] = -width/2
        basic_bgrid["ymax"] = +width/2
        basic_bgrid["zmin"] = -height/2
        basic_bgrid["zmax"] = +height/2
        basic_bgrid["origin"] = np.zeros(3)

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

def assemble_nbi_dict(config):

    print("     running 'assemble_nbi_dict' ...")

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

    nbi = {"name":"test_beam","shape":1,"data_source":'run_tests:test_beam',
           "src":uvw_src, "axis":uvw_axis, "widy":widy, "widz":widz,
           "divy":divy, "divz":divz, "focy":focy, "focz":focz,
           "naperture":naperture, "ashape":ashape, "adist":adist,
           "awidy":awidy, "awidz":awidz, "aoffy":aoffy, "aoffz":aoffz}

    return nbi
def create_fidasim_inputs_from_cql3dm(config, plot_flag, include_f4d):

    print("running 'create_fidasim_inputs_from_cql3dm' ...")

    # Define interpolation grid:
    rmin = config["rmin"]
    rmax = config["rmax"]
    nr = config["nr"]
    zmin = config["zmin"]
    zmax = config["zmax"]
    nz = config["nr"]
    grid = rz_grid(rmin,rmax,nr,zmin,zmax,nz)

    # Compute equil dict:
    # =====================
    file_name = config["equil_file_name"]
    equil, rho = read_fields(file_name,grid)

    # Compute the plasma dict:
    # ========================
    plasma = read_plasma(config,grid,rho,plot_flag=plot_flag)

    # Compute fbm dict for ions:
    # =========================
    # How to I enable reading both ion and electron f?
    # What about for multiple ion species?

    if include_f4d == True:
        fbm = read_f4d(config, grid, rho, plot_flag=plot_flag, interpolate_f4d=True)
    else:
        fbm = read_f4d(config, grid, rho, plot_flag=plot_flag, interpolate_f4d=False)

    # Compute nbi dict from cqlinput:
    # ===============================
    nbi = assemble_nbi_dict(config)

    # Compute inputs dict:
    # ===================
    inputs = assemble_inputs(config, nbi)

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

    # Plot results:
    # =============
    if (plot_flag):
        output_path = config["output_path"]

        fig = plt.figure(1)
        plt.contourf(grid["r2d"],grid["z2d"],equil["bz"].T)
        plt.colorbar
        plt.title("Bz_2D")
        plt.ylabel("Z [cm]")
        plt.xlabel("R [cm]")
        fig.set_size_inches(4, 6)  # Width, Height in inches
        plt.savefig(output_path + 'Bz_2D.png')
        # plt.show()

        fig = plt.figure(2)
        plt.contourf(grid["r2d"],grid["z2d"],equil["br"].T)
        plt.colorbar
        plt.title("Br_2D")
        plt.ylabel("Z [cm]")
        plt.xlabel("R [cm]")
        fig.set_size_inches(4, 6)  # Width, Height in inches
        plt.savefig(output_path + 'Br_2D.png')
        # plt.show()

        fig = plt.figure(3)
        plt.contour(grid["r2d"],grid["z2d"],rho,levels=np.linspace(0.01, 1.0, 16))
        plt.colorbar
        plt.title("flux_surfaces")
        plt.ylabel("Z [cm]")
        plt.xlabel("R [cm]")
        fig.set_size_inches(4, 6)  # Width, Height in inches
        plt.savefig(output_path + 'flux_surfaces.png')
        # plt.show()

        fig = plt.figure(4)
        plt.contourf(grid['r2d'], grid['z2d'], plasma['dene'].T*plasma['mask'].T)
        plt.contour(grid['r2d'], grid['z2d'], rho,levels=[1.0, 1.2, 1.3, 1.4])
        fig.set_size_inches(4, 6)  # Width, Height in inches

def read_preprocessor_config(file_name):

    # Read contents of the preprocessor configuration namelist file:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nml = f90nml.read(file_name)

    # Initialize user input dictionary:
    config = {}

    # Metadata for fidasim run:
    config["runid"] = nml['fidasim_run_info']['runid']
    config["comment"] = nml['fidasim_run_info']['comment']

    # R-Z Interpolation grid:
    sub_nml = nml['rz_interpolation_grid']
    config["rmin"] = sub_nml['rmin']
    config["rmax"] = sub_nml['rmax']
    config["nr"] = sub_nml['nr']
    config["zmin"] = sub_nml['zmin']
    config["zmax"] = sub_nml['zmax']
    config["nz"] = sub_nml['nz']

    # CQL3DM related files:
    # (Assume we only have a single ion species)
    sub_nml = nml['cql3d_input_files']
    input_dir = sub_nml['input_dir']
    config["equil_file_name"] = input_dir + sub_nml['equil_file_name']
    config["cqlinput"] = input_dir + sub_nml['cqlinput']
    config["plasma_file_name"] = input_dir + sub_nml['plasma_file_name']
    config["f4d_ion_file_name"] = input_dir + sub_nml['f4d_ion_file_name']
    config["f4d_electron_file_name"] = input_dir + sub_nml['f4d_electron_file_name']

    # Plasma profiles parameters:
    sub_nml = nml['plasma_profiles_params']
    config["impurity_charge"] = sub_nml['impurity_charge']
    config["rho_LCFS"] = sub_nml['rho_LCFS']
    config["dene_LCFS"] = sub_nml['dene_LCFS']
    config["te_LCFS"] = sub_nml['te_LCFS']
    config["ti_LCFS"] = sub_nml['ti_LCFS']
    config["zeff_LCFS"] = sub_nml['zeff_LCFS']

    # Define path to output directory:
    # (This is where you want the FIDASIM HDF5 files to be written to)
    config['output_path'] = nml['preprocessor_output']['output_dir']

    # Beam physics switches:
    sub_nml = nml['beam_physics_switches']
    if "enable_nonthermal_calc" in sub_nml:
        config['enable_nonthermal_calc'] = sub_nml['enable_nonthermal_calc']
    if "calc_sink" in sub_nml:
        config['calc_sink'] = sub_nml['calc_sink']

    config['calc_birth'] = sub_nml['calc_birth']
    config['calc_dcx'] = sub_nml['calc_dcx']
    config['calc_halo'] = sub_nml['calc_halo']

    # Define number of Monte-Carlo particles:
    sub_nml = nml['monte_carlo_particles']
    config['n_nbi'] = sub_nml['n_nbi']
    config['n_birth'] = sub_nml['n_birth']
    config['n_dcx'] = sub_nml['n_dcx']
    config['n_halo'] = sub_nml['n_halo']

    # Define beam grid parameters:
    sub_nml = nml['beam_grid']
    config["beam_grid"] = {}
    config["beam_grid"]["beam_aligned"] = sub_nml['beam_aligned']  # Option between machine-aligned or beam-aligned
    config["beam_grid"]["nx"] = sub_nml['nx']
    config["beam_grid"]["ny"] = sub_nml['ny']
    config["beam_grid"]["nz"] = sub_nml['nz']
    config["beam_grid"]["rstart"] = sub_nml['rstart']  # Only works for "beam_aligned" == True
    config["beam_grid"]["length"] = sub_nml['length']
    config["beam_grid"]["width"] = sub_nml['width']
    config["beam_grid"]["height"] = sub_nml['height']

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

def main():
    print("Add some code")

if __name__=='__main__':
    main()