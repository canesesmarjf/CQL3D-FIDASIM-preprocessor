import sys
import os
import warnings
import f90nml
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

# Add FIDASIM specific libraries
FIDASIM_dir = "/home/jfcm/Repos/FIDASIM/"
sys.path.append(FIDASIM_dir + 'lib/python')

from fidasim.utils import read_geqdsk, rz_grid, beam_grid, read_ncdf
import fidasim as fs
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode

# Define physical constans:
mass_proton_CGI = 1.6726E-24 # CGI units [g]
charge_electron_SI = 1.60217663E-19  # SI units [C]
mass_proton_SI = 1.67262192E-27  # SI units [Kg]
mass_electron_SI = 9.1093837E-31  # SI units [Kg]

def read_fields(file_name,grid):
    fields, rho, btipsign = read_geqdsk(file_name, grid, poloidal=True)
    return fields, rho
def read_plasma(user_input,grid,rho,plot_flag=False):

    # Read CQL3D standard .nc file into dict:
    nc_data = read_ncdf(user_input["plasma_file_name"])

    # Charge number of main impurity species:
    impurity_charge = user_input["impurity_charge"]

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
    # Electron density profile from nc file, dimensions [tdim, r0dim, zdim, species]
    dene_nc = nc_data['densz1'][-1,:,:,1]

    # Symmetrize it about the R axis:
    flipped_dene_nc = np.flip(dene_nc, axis=1)
    dene_nc = np.concatenate([flipped_dene_nc, dene_nc[:, 1:]], axis=1)

    # ELECTRON TEMPERATURE:
    # =====================
    # Electron temperature profile from nc file, dimensions [tdim, r0dim, species, zdim]
    te_nc = nc_data['energyz'][-1,:,1,:]

    # Symmetrize it about the R axis:
    flipped_te_nc = np.flip(te_nc, axis=1)
    te_nc = np.concatenate([flipped_te_nc, te_nc[:, 1:]], axis=1)

    # ION TEMPERATURE:
    # ==================
    # Ion temperature profile from nc file, dimensions [tdim, r0dim, species, zdim]
    ti_nc = nc_data['energyz'][-1,:,0,:]

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
    dene_LCFS = user_input["dene_LCFS"]
    te_LCFS = user_input["te_LCFS"]
    ti_LCFS = user_input["ti_LCFS"]
    zeff_LCFS = user_input["zeff_LCFS"]
    rho_LCFS = user_input["rho_LCFS"]

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
    plasma = {"time":nc_data['time'][-1], "data_source":user_input["plasma_file_name"], "mask":mask,
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
        plt.savefig(user_input["output_path"] + 'dene.png')

        fig = plt.figure(6)
        plt.contourf(r2d, z2d, te * mask)
        plt.colorbar()
        plt.plot(r_nc.T, z_nc.T, color='r', lw=1.0)
        plt.xlim([0, 25])
        plt.xlabel('R [cm]')
        plt.title('Electron temperature [keV]')
        fig.set_size_inches(4, 6)  # Width, Height in inches
        plt.savefig(user_input["output_path"] + 'te.png')

        fig = plt.figure(7)
        plt.contourf(r2d,z2d,ti*mask)
        plt.colorbar()
        plt.plot(r_nc.T,z_nc.T,color='r',lw=1.0)
        plt.xlim([0,25])
        plt.xlabel('R [cm]')
        plt.title('Ion temperature [keV]')
        fig.set_size_inches(4, 6)  # Width, Height in inches
        plt.savefig(user_input["output_path"] + 'ti.png')

    return plasma
def read_f4d(user_input,grid,rho,plot_flag=False):

    # Physical constants in SI units:
    # charge_electron_SI = 1.60217663E-19  # [C]
    # mass_proton_SI = 1.67262192E-27  # [Kg]
    # mass_electron_SI = 9.1093837E-31  # [Kg]

    # Read nc file into dict:
    f4d_nc = read_ncdf(user_input['f4d_ion_file_name'])

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

    # Interpolation grids:
    r_grid = grid['r2d']
    z_grid = grid['z2d']

    # Create query points from interpolation grid, has dimensions [Rg*Zg,2]
    query_points = np.array([r_grid.flatten(),z_grid.flatten()]).T

    # Perform interpolation, has dimensions [Rg*Zg,E*P]
    interpolated_values = interpolator(query_points)

    # Reshape interpolated values back into the grid shape: [Rg,Zg,E,P]
    fbm_grid = interpolated_values.reshape(r_grid.shape[0], z_grid.shape[1], fbm_nc.shape[0], fbm_nc.shape[1])

    # Reorder dimensions back to original [E,P,Rg,Zg]
    fbm_grid = fbm_grid.transpose(2, 3, 0, 1)

    # Apply mask to fbm_grid:
    mask_reshaped = mask[np.newaxis,np.newaxis,:,:]
    fbm_grid *= mask_reshaped

    # Calculating fbm density:
    # =========================
    denf = np.zeros(z_grid.shape)

    # Integrate over velocity (v) and pitch (v):
    p = np.cos(t_nc)
    v = v_nc
    int_over_v = np.trapz(fbm_grid * v[:, np.newaxis, np.newaxis, np.newaxis] ** 2, x=v, axis=0)
    denf = -2 * np.pi * np.trapz(int_over_v, x=p, axis=0)

    # Get time stamp and mass for this dataset:
    # byte_data = f4d_nc['mnemonic'].tobytes()
    # decoded_string = byte_data.decode('utf-8')
    # clean_string = decoded_string.strip('\x00').strip()
    # tokens = user_input['f4d_ion_file_name'].split('/')
    # source_file_name = tokens[0] + '/' + tokens[1] + '/' +  clean_string + ".nc"
    # src_nc = read_ncdf(source_file_name)
    src_nc = read_ncdf(user_input['plasma_file_name'])
    time = src_nc['time'][-1]
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

    fbm_dict = {"type":1,"time":time,"nenergy":nenergy,"energy":energy,"npitch":npitch,
              "pitch":pitch,"f":fbm_grid,"denf":denf,"data_source":os.path.abspath(user_input['f4d_ion_file_name'])}

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
        iz = 60
        ir = 0
        data = fbm_nc[:, :, ir, iz]
        log_data = np.where(data > 0, np.log10(data), np.nan)
        contours = np.logspace(13, np.nanmax(log_data) * 1.1, 32)
        plt.contour(vvpar * vnc, vvper * vnc, data, colors='black', levels=contours)
        plt.contourf(vvpar * vnc, vvper * vnc, log_data, levels=np.log10(contours), cmap='viridis')
        plt.colorbar()
        plt.axis('image')
        plt.xlim([-0.01, 0.01])
        plt.ylim([0, 0.01])
        plt.title('Ion distribution function, R = 0, Z = 0')
        plt.savefig(user_input['output_path'] + 'f4d_ion_vpar_vper.png')

        # Plot #2: Compare input and interpolated f4d:
        # ============================================

        # Create a figure and two subplots, side by side (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Input f4d:
        ir = 0
        iz = 50
        data = fbm_grid[:, :, ir, iz]
        ax1.contourf(ee_nc, pp_nc, data)
        ax1.set_xlim([0, enorm])
        ax1.set_xlabel('Energy [keV]')
        ax1.set_ylabel('pitch $v_{\parallel}/v$')
        ax1.set_title('INPUT: Ion f4d, R = 0, Z = 0')

        # Output f4d:
        ir = 0
        iz = 60
        data = fbm_nc[:,:,ir,iz]
        ax2.contourf(ee_nc,pp_nc,data)
        ax2.set_xlim([0,enorm])
        ax2.set_xlabel('Energy [keV]')
        ax2.set_title('OUTPUT: Ion f4d, R = 0, Z = 0')

        # Save figure:
        fig.savefig(user_input['output_path'] + 'f4d_ion_energy_pitch.png')

        # Plot #3: Compare ion density from integral:
        # ===========================================

        # Get density from source file:
        r_src = src_nc['solrz']
        z_src = src_nc['solzz']
        dene_src = src_nc['densz1'][-1,:,:,0]

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
        fig.savefig(user_input['output_path'] + 'density_f4d_comparison.png')

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
        fig.savefig(user_input['output_path'] + 'density_zprof_f4d_comparison.png')

        # Plot #5: Directly compare R profiles:
        # ===================================

        # Create figure object:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(wspace=0.5)  # Adjust the width space between subplots

        ax1.plot(r_grid[:, 50], denf[:,50], color='red', lw=4, label='f4d')
        ax1.plot(r_src[:, 0], dene_src[:,0], color='black', lw=2, label='src')
        ax1.set_xlim([0, 15])
        ax1.set_ylabel("density [cm^{-3}]")
        ax1.set_xlabel("Z [cm]")
        ax1.set_yscale('log')
        ax1.set_title('log scale: R profile')
        ax1.legend()

        ax2.plot(r_grid[:, 50], denf[:,50], color='red', lw=4, label='f4d')
        ax2.plot(r_src[:, 0], dene_src[:,0], color='black', lw=2, label='src')
        ax2.set_xlim([0, 15])
        ax2.set_ylabel("density [cm^{-3}]")
        ax2.set_xlabel("Z [cm]")
        ax2.set_yscale('linear')
        ax2.set_title('lin scale: R profile')
        ax1.legend()

        # Save figure:
        fig.savefig(user_input['output_path'] + 'density_rprof_f4d_comparison.png')

    return fbm_dict
def assemble_inputs(user_input, nbi):

    # Extract paths:
    cqlinput = user_input["cqlinput"]
    nc_file_name = user_input["plasma_file_name"]
    output_path = user_input["output_path"]

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
    time = nc['time'][-1]

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
    calc_dcx = user_input["calc_dcx"]
    calc_halo = user_input["calc_halo"]
    calc_cold = 0
    calc_birth = user_input["calc_birth"]
    calc_fida_wght = 0
    calc_npa_wght = 0
    calc_pfida = 0
    calc_pnpa = 0

    # Number of particles to track:
    n_nbi = int(user_input["n_nbi"])
    n_halo = int(user_input["n_halo"])
    n_dcx = int(user_input["n_dcx"])
    n_birth = int(user_input["n_birth"])
    n_fida = int(1e3)
    n_npa = int(1e3)
    n_pfida = int(1e3)
    n_pnpa = int(1e3)

    basic_inputs = {"seed":seed, "device":"test", "shot":1, "time":time,
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
    nx = user_input["beam_grid"]["nx"]
    ny = user_input["beam_grid"]["ny"]
    nz = user_input["beam_grid"]["nz"]
    rstart = float(user_input["beam_grid"]["rstart"])
    length = float(user_input["beam_grid"]["length"])
    width = float(user_input["beam_grid"]["width"])
    height = float(user_input["beam_grid"]["height"])

    if (user_input["beam_grid"]["beam_aligned"]):
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

    # Metadata on simulation run:
    inputs["comment"] = user_input["comment"]
    inputs["runid"] = user_input["runid"]

    return inputs
def assemble_nbi_dict(user_input):

    # Read contents of cqlinput namelist file:
    # Read in namelist:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nml = f90nml.read(user_input['cqlinput'])

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
def run_prefida(user_input, plot_flag):

    # Define interpolation grid:
    rmin = user_input["rmin"]
    rmax = user_input["rmax"]
    nr = user_input["nr"]
    zmin = user_input["zmin"]
    zmax = user_input["zmax"]
    nz = user_input["nr"]
    grid = rz_grid(rmin,rmax,nr,zmin,zmax,nz)

    # Compute equil dict:
    # =====================
    file_name = user_input["equil_file_name"]
    equil, rho = read_fields(file_name,grid)

    # Compute the plasma dict:
    # ========================
    plasma = read_plasma(user_input,grid,rho,plot_flag=plot_flag)

    # Compute fbm dict for ions:
    # =========================
    fbm = read_f4d(user_input,grid,rho,plot_flag=plot_flag)

    # Compute nbi dict from cqlinput:
    # ===============================
    nbi = assemble_nbi_dict(user_input)

    # Compute inputs dict:
    # ===================
    inputs = assemble_inputs(user_input, nbi)

    # Produce input files for FIDASIM:
    # ================================
    fs.prefida(inputs, grid, nbi, plasma, equil, fbm)

    # Plot results:
    # =============
    output_path = user_input["output_path"]

    if (plot_flag):
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

def main():
    # Initialize user input dictionary:
    user_input = {}

    # Metadata for run:
    user_input["runid"] = "WHAM_example_debug"
    user_input["comment"] = "FIDASIM NBI thermal deposition in WHAM with halo physics enabled"

    # R-Z Interpolation grid:
    user_input["rmin"] = 0.01
    user_input["rmax"] = 60
    user_input["nr"] = 100
    user_input["zmin"] = -200
    user_input["zmax"] = +200
    user_input["nz"] = 100

    # CQL3DM related files:
    # (Assume we only have a single ion species)
    user_input["equil_file_name"] = "./Step_1_input/eqdsk_wham_expander"
    user_input["cqlinput"] = "./Step_1_input/cqlinput"
    user_input["plasma_file_name"] = "./Step_1_input/WHAM2expander_NB100_nitr40npz2_iter0_sh09_iy300jx300lz120.nc"
    user_input["f4d_ion_file_name"] = "./Step_1_input/WHAM2expander_NB100_nitr40npz2_iter0_sh09_iy300jx300lz120_f4d_001.nc"
    user_input["f4d_electron_file_name"] = "./Step_1_input/WHAM2expander_NB100_nitr40npz2_iter0_sh09_iy300jx300lz120_f4d_002.nc"

    # Plasma profiles parameters:
    user_input["impurity_charge"] = 6
    user_input["rho_LCFS"] = 1.0
    user_input["dene_LCFS"] = 1e9
    user_input["te_LCFS"] = 0.005
    user_input["ti_LCFS"] = 0.005
    user_input["zeff_LCFS"] = 1.0

    # Define path to output directory:
    # (This is where you want the FIDASIM HDF5 files to be written to)
    user_input['output_path'] = "./Step_1_output_debug/"

    # Beam physics switches:
    user_input['calc_birth'] = 1
    user_input['calc_dcx'] = 1
    user_input['calc_halo'] = 1

    # Define number of Monte-Carlo particles:
    user_input['n_nbi'] = 1e3
    user_input['n_birth'] = 1e3
    user_input['n_dcx'] = 1e5
    user_input['n_halo'] = 1e3

    # Define beam grid parameters:
    user_input["beam_grid"] = {}
    user_input["beam_grid"]["beam_aligned"] = False # Option between machine-aligned or beam-aligned
    user_input["beam_grid"]["nx"] = 70
    user_input["beam_grid"]["ny"] = 50
    user_input["beam_grid"]["nz"] = 50
    user_input["beam_grid"]["rstart"] = 60 # Only works for "beam_aligned" == True
    user_input["beam_grid"]["length"] = 150
    user_input["beam_grid"]["width"] = 80
    user_input["beam_grid"]["height"] = 130

    # Create FIDASIM input files using prefida function:
    plot_flag = True
    run_prefida(user_input,plot_flag)
    print("End of script")

if __name__=='__main__':

    # Run main function:
    main()
