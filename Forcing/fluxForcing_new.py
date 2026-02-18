import re
import os
import glob
import subprocess
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import gsw
import xarray as xr  #used in get_latlon()

from netCDF4 import Dataset
FORCING_FOLDER = os.path.dirname(__file__)
ROOT = os.path.dirname(FORCING_FOLDER)
RUN_FOLDER = os.path.join(ROOT, 'Run')
EXTERNAL_FOLDER = os.path.join(ROOT, 'External')
INCLUDE_FOLDER = os.path.join(ROOT, 'Include')
RESULT_FOLDER = os.path.join(ROOT, 'Results')

SEC_PER_DAY = 24.0 * 3600.0
NANOSEC_PER_SECOND = 1e9

def now():
    """Returns current date time on isoformat: yyyy-mm-ddThhmmss"""
    return datetime.now().isoformat("T", "seconds").replace(':', '')


def extract_experiment_number(filename_string):
    """
    Extracts the digits following '_exp' in a ROMS simulation string.
    """
    # Search for '_exp' followed by one or more digits (\d+)
    match = re.search(r'_exp(\d+)', filename_string)
    
    if match:
        return int(match.group(1))
    
    return None


def print_forcing_info(filename, names=None):
    f = Dataset(filename, 'r')
    if names is None:
        names = ['swrad', 'shflux', 'swflux', 'sustr', 'svstr']
    for name in names:
        var = f.variables[name]
        avg = np.mean(var)
        vmin = np.min(var)
        vmax  = np.max(var)
        print(name, vmin, avg, vmax)
        print('')

def make_bulkforce_file(diurnal=True):
    """Generate bulk forcing file"""
    # Generate empty forcing file and open for editing
    os.system('ncgen -b roms_bulkforce.cdl')
    f = Dataset(os.path.join(FORCING_FOLDER,'roms_bulkforce.nc'), 'a')

    # Make list of forcing variables to set (must match "roms_bulkfrc.cdl"!)
    frcvar = ['lwrad_down','cloud', 'Uwind', 'Vwind', 'Pair', 'Tair', 'Qair', 'rain', 'swrad']

    # Loop over variables and initialize to zero.
    #
    # NOTE! For more advanced input, edit yourself.
    #
    for varname in frcvar:
        print(varname)
        var = f.variables[varname]
        var[:] = 0.0

    # Setting lwrad_down to 400
    #f.variables['lwrad_down'][:,:,:] = 400.0

    # Setting swrad to constant value # See below for variable swrad
    #f.variables['swrad'][:,:,:] = 300.0    

    # Setting constant wind in x-direction
    f.variables['Uwind'][:,:,:] = 5.0

    # Setting cloud cover to 1
    f.variables['cloud'][:,:,:] = 0.0

    # Setting constant humidity
    f.variables['Qair'][:,:,:] = 80.0

    # Setting constant temperature
    f.variables['Tair'][:,:,:] = 20.0 #10.0 old

    # Setting constant air pressure
    f.variables['Pair'][:,:,:] = 1020.0

    # Set time variable, the end time is the important one.
    #
    # NOTE! If time varying input, which requires more than two
    # time steps, remember to edit the size of "ocean_time" at
    # the start of "roms_frc.cdl".
    #
    ot = f.variables['ocean_time']
    # # ot.units == 'modified Julian day'
    num_time_steps = len(ot) #169
    #timevec = np.linspace(0.0,7*24*3600.0,num_time_steps) 
    timevec = np.linspace(0.0,7.0,num_time_steps)
    ot[:] = timevec

    # Setting shortwave diurnal cycle
    swrad = np.zeros_like(f.variables['swrad'][:])
    onedayfreq = 2*np.pi
    #sw_amplitude = -300.0  # Old
    sw_amplitude = 800.0  # new
    # Antar at timevec er i DAGER. Faseforskyvning på 0.5 for å ha toppen ved middag (0.5 dager)
    phase_shift = 0.5

    if diurnal:
        for i in range(num_time_steps):
            swrad_value = sw_amplitude * np.cos(onedayfreq * (timevec[i] - phase_shift))
            # Setter negative verdier til null (ingen sol om natten)
            swrad_value = max(0.0, swrad_value) 

            swrad[i,:,:] = swrad_value * np.ones((14,12)) # 14x12 er antall celler i ditt grid
    else: #average diurnal
        swrad = np.maximum(sw_amplitude * np.cos(onedayfreq * (timevec - phase_shift)), 0).mean()
        print(f"Mean swrad: {swrad}")
        

    f.variables['swrad'][:,:,:] = swrad

    # Sync file to force write, then close.
    f.sync()
    f.close()
    # Rename bulk forcing file
    os.system('mv {0}/roms_bulkforce.nc {0}/roms_frc.nc'.format(FORCING_FOLDER))


def make_flux_force_file():
    """Generate flux forcing file"""
    # Generate empty forcing file and open for editing
    os.system('ncgen -b roms_flux.cdl')
    f = Dataset(os.path.join(FORCING_FOLDER, 'roms_flux.nc'), 'a')

    # Make list of forcing variables to set (must match "roms_frc.cdl"!)
    frcvar = ['swrad', 'shflux', 'swflux', 'sustr', 'svstr']

    # Loop over variables and initialize to zero.
    #
    # NOTE! For more advanced input, edit yourself.
    #
    for varname in frcvar:
        print(varname)
        var = f.variables[varname]
        var[:] = 0.0

    # Setting constant stress in x-direction  Typical range [0, 0.5]
    sustr = f.variables['sustr']
    #sustr[:] = 0.0
    #sustr[1] = 0.5
    sustr[:] = 0.1 

    # Setting constant cooling
    # sensible heat flux range [-100, 100]
    # Negative sign means flux coming out of the sea
    # Positive sign means flux into the sea
    shflux = f.variables['shflux']  
    #shflux[0:2] = 0.0
    #shflux[2:8] = 100.0
    #shflux[:] = -300.0
    shflux[:] = 100.0

    # Set time variable, the end time is the important one.
    #
    # NOTE! If time varying input, which requires more than two
    # time steps, remember to edit the size of "ocean_time" at
    # the start of "roms_frc.cdl".
    #
    ot = f.variables['ocean_time']
    # ot.units == 'seconds since 1970-01-01'
    #ot[:] = [0.0,7*24*3600.0] 
    num_time_steps = 169
    timevec = np.linspace(0.0, 7*24*3600.0, num_time_steps) 
    ot[:] = timevec

    # Setting shortwave diurnal cycle
    swrad = np.zeros_like(f.variables['swrad'][:])
    onedayfreq = 2*np.pi/(3600.*24)
    #sw_amplitude = -0.0  # Old
    sw_amplitude = 500.0  # new
    for i in range(num_time_steps):
        swrad[i,:,:] = sw_amplitude*np.cos(onedayfreq*timevec[i])*np.ones((14,12))
        
    swrad[np.where(swrad<0)] = 0.0


    f.variables['swrad'][:,:,:] = swrad

    # Sync file to force write, then close.
    f.sync()
    f.close()
    # Rename flux forcing file
    os.system('mv {0}/roms_flux.nc {0}/roms_frc.nc'.format(FORCING_FOLDER))


def run_roms(folder):
    """
    Parameters
    ----------
    folder: str
        Folder to store the results from the run
    
    Notes
    -----
    Run current compiled ROMS model using the flux forcing file 
    and saves the results to the given result folder.
    The build roms script is also copied to the given result folder.
    """
    # Copy flux forcing file to Run folder
    txt = subprocess.run(['cp', 
                          os.path.join(FORCING_FOLDER,'roms_frc.nc'),
                          os.path.join(RUN_FOLDER,'roms_frc.nc')], 
                         capture_output=True, text=True).stdout
    print("Finished generating forcing file: roms_frc.nc", txt)

    #Run romsS
    #os.system("../Run/romsS < ../External/roms_column.in > roms.log")
    os.system("{}/romsS < {}/roms_column.in > {}/roms.log".format(RUN_FOLDER, 
                                                                  EXTERNAL_FOLDER, 
                                                                  FORCING_FOLDER))
    print("Finished running romsS")
    move_results(folder)


def move_results(folder):
    print('Move input and results to ', folder)
    os.system('mkdir ' + folder)
    # log = subprocess.run(['mkdir', folder], capture_output=True, text=True).stdout
    print("Finished making folder:", folder)
    for fname in ['roms.log', 'roms_frc.nc', 'roms_his.nc', 'roms_rst.nc', ]:
        log = subprocess.run(['mv', 
                              os.path.join(FORCING_FOLDER, fname), 
                              os.path.join(folder, fname)], 
                              capture_output=True, text=True).stdout
        print('Moved ', fname)
    for fname in ['build_roms.sh']: 
        subprocess.run(['cp', 
                        os.path.join(ROOT, fname), 
                        os.path.join(folder, fname)], 
                        capture_output=True, text=True).stdout
        print('Copied ', fname)
    
    for subfolder in ['External','Include']:
        in_folder = os.path.join(ROOT, subfolder)
        out_folder = os.path.join(folder, subfolder)
        os.system('mkdir ' + out_folder)
        for fname in os.listdir(in_folder):
            subprocess.run(['cp', 
                        os.path.join(in_folder,fname), 
                        os.path.join(out_folder,fname)], 
                        capture_output=True, text=True).stdout
            print('Copied ', fname, ' to ', subfolder)
    

    #roms_log = subprocess.run(['../Run/romsS', '../External/roms_column.in'], capture_output=True, text=True).stdout
    #print("Finished running romsS:", roms_log)


def plot_density_hovmuller(romsfile, maxdensity, filename=None):
    """Open history files and plot Hovmuller density profiles"""
    f = Dataset(romsfile, 'r')

    # Get vertical coordinate
    z_r = f.variables['z_rho'][:,:,7,6]

    # Get density 
    rho = f.variables['rho'][:,:,7,6]

    # Get time
    otime = f.variables['ocean_time'][:]
    otime = otime/(24*3600.)
    ny = np.shape(rho)[1]
    dt = np.array([otime,]*ny).transpose()

    # Check if we should limit range
    if maxdensity > 0.0:
        rho[rho>maxdensity] = np.nan

    # Open figure
    
    plt.figure()
    print(np.shape(dt))
    print(np.shape(rho))
    print(np.shape(z_r))

    # Plot filled contours
    plt.contourf(dt, z_r, rho, 200, cmap=cm.ocean_r)

    # Add info 
    plt.colorbar(label='Potential density anomaly [kg/m^3]')
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    if filename:
        plt.savefig(filename)
    # plt.show()

    # Close file
    f.close()


def plot_speed_hovmuller(romsfile, filename=None):
    """Open history files and plot Hovmuller speed profiles"""
    # Open history files and plot profiles
    f = Dataset(romsfile, 'r')

    # Get vertical coordinate
    z_r = f.variables['z_rho'][:,:,7,6]

    # Get u,v
    u = f.variables['u'][:,:,7,6]  # N X M
    v = f.variables['v'][:,:,7,6]  # N X M 
    speed = np.sqrt(u**2 + v**2)
    # speed[speed>0.2] = 0.2  # LJB

    # Get time
    otime = f.variables['ocean_time'][:]
    otime = otime/(24*3600.)
    # dt = np.array([otime,]*42).transpose()  # LJB
    n = np.shape(speed)[1]
    dt = np.array([otime,]*n).transpose()  # LJB

    # Open figure
    plt.figure()

    # Plot filled contours
    #plt.contourf(dt,z_r,speed,levels=np.logspace(-6,0,25),norm=colors.LogNorm())
    # plt.contourf(dt,z_r,speed,25)  # LJB
    plt.contourf(dt,z_r,speed,50)  # LJB

    # Add info 
    plt.colorbar(label='Speed [m/s]')
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    if filename:
        plt.savefig(filename)
    # plt.show()

    # Close file
    f.close()


def plot_hodograph(romsfile, savefile=False, ext='.png'):
    # Check how many input files
    N = len(romsfile)

    # Plot hodograph
    plt.figure()

    for i in range(N):
        # Open history files and plot profiles
        f = Dataset(romsfile[i], 'r')

        # Get velocities at rho points
        u = f.variables['u_eastward'][:,-1,7,6]
        v = f.variables['v_northward'][:,-1,7,6]

        dt = 3600 # Output time step
        plt.plot([0],[0],'rd')
        
        if i==0:
            plt.plot(u, v, 'b-')
        else:
            plt.plot(u, v, 'g-')

        plt.axis('equal')
        if savefile:
            folder = os.path.dirname(romsfile[i])
            plt.savefig(os.path.join(folder, 'hodograph' +ext))
        # plt.show()

        # Close file
        f.close()


def get_latlon(i,j):
    

    # Open the ROMS grid file
    grid_ds = xr.open_dataset('roms_grd.nc')

    # Access the latitude and longitude arrays
    latitudes = grid_ds['lat_rho']
    longitudes = grid_ds['lon_rho']

    # Use the indices 7 and 6 to get the specific location
    # The dimensions are typically (eta_rho, xi_rho), corresponding to latitude and longitude
    lat_value = latitudes[i, j]
    lon_value = longitudes[i, j]

    print(f"Latitude: {lat_value.values}")
    print(f"Longitude: {lon_value.values}")


def get_density(file_path='roms_his.nc'):

    ds = xr.open_dataset(file_path)
    # Velg første tidspunkt og midtre lag for tetthet
    rho_data = ds['rho'].isel(ocean_time=0, s_rho=len(ds['s_rho']) // 2)


def calculate_roms_cell_volume(file_path: str):
    """
    Calculates the volume of each grid cell in a ROMS history file.

    The volume of a 3D ROMS grid cell is calculated by multiplying its
    horizontal area by its vertical thickness.

    Formula: Volume = (1 / pm) * (1 / pn) * Hz

    Where:
    - (1/pm) and (1/pn) are the grid spacing in the xi and eta directions.
    - pm and pn are the curvilinear coordinate metrics.
    - Hz is the vertical layer thickness at RHO-points.

    Args:
        file_path (str): The path to the ROMS history NetCDF file (e.g., 'roms_his.nc').
    """
    try:
        # Load the ROMS history file using xarray
        print(f"Loading dataset from: {file_path}")
        ds = xr.open_dataset(file_path)

        # Print a summary of the dataset to verify variables are present
        print("\nDataset loaded successfully. Available variables:")
        print(ds)

        # Check for the required variables. We now handle the case where Hz is missing.
        required_vars = ['pm', 'pn', 'z_w']
        for var in required_vars:
            if var not in ds.data_vars:
                raise KeyError(f"Required variable '{var}' not found in the dataset. "
                               "Please ensure it is included in your ROMS output.")

        # Extract the necessary variables. xarray automatically handles dimensions.
        pm = ds['pm']
        pn = ds['pn']
        if 'Hz' in ds:
            Hz = ds['Hz']
        else:
            z_w = ds['z_w']

            # If Hz is not in the file, calculate it from z_w.
            # z_w has N+1 vertical levels (from k=0 to N), while z_rho has N levels (k=1 to N).
            # The thickness of a rho layer is the difference between the z_w faces that bound it.
            print("\n'Hz' variable not found. Calculating vertical layer thickness (Hz) from 'z_w'...")
            # The slicing correctly aligns the dimensions for subtraction.
            Hz = z_w.diff(dim='s_w')
        
            # The new dimension 's_w' from the slicing now corresponds to 's_rho'.
            # We rename the dimension here to avoid the "cannot add coordinates" error.
            Hz = Hz.rename({'s_w': 's_rho'})

        # The core calculation: Volume = Horizontal Area * Vertical Thickness
        print("\nCalculating grid cell volume...")
        volume = (1 / pm) * (1 / pn) * Hz

        print("\nCalculation complete.")
        
        # Get the dimensions for robust indexing
        time_size = volume.sizes.get('ocean_time', 1)
        s_rho_size = volume.sizes.get('s_rho', 1)
        eta_rho_size = volume.sizes.get('eta_rho', 1)
        xi_rho_size = volume.sizes.get('xi_rho', 1)
        
        print(f"Shape of the resulting 'volume' array: ({time_size}, {s_rho_size}, {eta_rho_size}, {xi_rho_size})")

        # Let's inspect the data at a specific point for verification
        # Use robust indices to prevent out-of-bounds errors
        time_step = 0
        s_layer = s_rho_size // 2  # Middle layer
        eta_index = eta_rho_size // 2 # Middle of the eta dimension
        xi_index = xi_rho_size // 2  # Middle of the xi dimension

        print(f"\n--- Sample Data at Time Step {time_step}, Layer {s_layer}, Grid Cell ({xi_index}, {eta_index}) ---")
        
        # Select the sample data point
        sample_volume = volume.isel(ocean_time=time_step, s_rho=s_layer, eta_rho=eta_index, xi_rho=xi_index).values
        sample_pm = pm.isel(eta_rho=eta_index, xi_rho=xi_index).values
        sample_pn = pn.isel(eta_rho=eta_index, xi_rho=xi_index).values
        sample_Hz = Hz.isel(ocean_time=time_step, s_rho=s_layer, eta_rho=eta_index, xi_rho=xi_index).values

        print(f"pm: {sample_pm} (1/m)")
        print(f"pn: {sample_pn} (1/m)")
        print(f"Hz: {sample_Hz} (m)")
        print(f"Calculated volume: {sample_volume} (m^3)")

        # You can now save the volume to a new NetCDF file or plot it.
        # Example of saving the calculated volume to a new file:
        # volume.to_netcdf('roms_cell_volume.nc')

        # Example of a simple visualization for the first time step, mid-layer
        print("\nCreating a visualization of the cell volume for the first time step, mid-layer...")
        plt.figure(figsize=(10, 8))
        mid_layer_volume = volume.isel(ocean_time=time_step, s_rho=s_layer)
        mid_layer_volume.plot(x='xi_rho', y='eta_rho')
        plt.title(f'ROMS Grid Cell Volume at Layer {s_layer} (Time Step {time_step})')
        plt.xlabel('xi_rho')
        plt.ylabel('eta_rho')
        plt.show()

        print("\nScript finished successfully.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
    except KeyError as e:
        print(f"Error: A required variable is missing. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def robust_time_conversion(time_da: xr.DataArray):
    """
    Converts Xarray DataArray time to float (seconds and days) based on the 'units' attribute.
    
    Returns:
        tuple: (time_float_sec, time_days_da) 
               where time_float_sec is time in seconds (float)
               and time_days_da is a DataArray with time in days, used for coordinates.
    """
    

    time_values = time_da.values
    units = time_da.attrs.get('units', '').lower()
    
    # Case 1: Time is numeric and in DAYS (e.g., modified julian day)
    if 'day' in units and np.issubdtype(time_values.dtype, np.number):
        time_days_values = time_values.astype(float)
        time_float_sec = time_days_values * SEC_PER_DAY
        
    # Case 2: Time is numeric, assumed to be SECONDS
    elif np.issubdtype(time_values.dtype, np.number):
        time_float_sec = time_values.astype(float)
        time_days_values = time_float_sec / SEC_PER_DAY
        
    # Case 3: Time is datetime-like (e.g., datetime64[ns])
    else:
        # Convert nanoseconds to seconds
        time_float_sec = time_values.astype('int64').astype(float) / NANOSEC_PER_SECOND
        time_days_values = time_float_sec / SEC_PER_DAY

    # Create a DataArray with 'days' as coordinate for robust interpolation
    time_days_da = xr.DataArray(
        time_days_values,
        dims=time_da.dims,
        coords={time_da.dims[0]: time_days_values}
    )
    time_days_da.attrs['units'] = 'days'
    return time_float_sec, time_days_da


def plot_thermodynamic_fluxes(his_file: str, frc_file: str, filename: str=''):
    """
    Loads ROMS history and forcing files and plots the four main components 
    of the thermodynamic heat flux ($Q_H$) over time, averaged over the entire grid (W/m^2).
    
    The equation represented is:
    $Q_H = Q_{SW} + Q_{LW}^{\\text{net}} + Q_S + Q_L$
    
    The components plotted are:
    1. Shortwave (Q_SW)
    2. Net Longwave (Q_LW_net = Q_LW_down - Q_LW_out)
    3. Sensible Heat Flux (Q_S)
    4. Latent Heat Flux (Q_L)
    
    If Q_S and Q_L are not available separately, the Net Turbulent Flux (shflux) 
    is plotted as a single component instead.
    """
    try:
        print(f"Loading history file for SST: {his_file}")
        ds_his = xr.open_dataset(his_file)
        print(f"Loading forcing file for input data: {frc_file}")
        ds_frc = xr.open_dataset(frc_file)

        # --- TIME HANDLING ---
        _, his_time_days = robust_time_conversion(ds_his['ocean_time'])
        _, frc_time_days = robust_time_conversion(ds_frc['ocean_time'])
        
        # --- 1. Outgoing Longwave Radiation (Q_LW_out) Calculation ---
        epsilon = 0.97  # Emissivity
        sigma = 5.67e-8 # Stefan-Boltzmann constant (W m^-2 K^-4)

        # Retrieve Sea Surface Temperature (SST) in Kelvin
        sst_celsius = ds_his['temp'].isel(s_rho=-1, drop=True)
        sst_kelvin = sst_celsius + 273.15
        
        q_lw_out = epsilon * sigma * sst_kelvin**4
        q_lw_out = q_lw_out.assign_coords(ocean_time=his_time_days)
        
        # Interpolate Q_LW_out to forcing timesteps for calculation consistency
        q_lw_out_interp = q_lw_out.interp(
            ocean_time=frc_time_days.values, 
            method='linear', 
            kwargs={"fill_value": "extrapolate"}
        ).mean(dim=['eta_rho', 'xi_rho']) # Spatial average
        
        # --- 2. Main Components from Forcing File (Spatially Averaged) ---
        
        # Shortwave Radiation (Q_SW) - Positive = Heating
        q_sw = ds_frc['swrad'].mean(dim=['eta_rho', 'xi_rho'])
        q_sw = q_sw.assign_coords(ocean_time=frc_time_days)
        
        # Downward Longwave Radiation (Q_LW_down) - Positive = Heating
        q_lw_down = ds_frc['lwrad_down'].mean(dim=['eta_rho', 'xi_rho'])
        q_lw_down = q_lw_down.assign_coords(ocean_time=frc_time_days)
        
        # Net Longwave Flux (Q_LW_net) - Can be positive or negative
        q_lw_net = q_lw_down - q_lw_out_interp
        q_lw_net.name = 'Q_LW_net'

        # --- 3. Turbulent Flux Components (Q_S and Q_L) ---
        q_sensible = None
        q_latent = None
        q_turb_net = xr.zeros_like(q_sw) # Default Net Turbulent Flux
        
        turbulent_components_available = False
        
        if 'sensible' in ds_frc.variables and 'latent' in ds_frc.variables:
            # Separate Sensible and Latent Fluxes are available
            q_sensible = ds_frc['sensible'].mean(dim=['eta_rho', 'xi_rho'])
            q_sensible = q_sensible.assign_coords(ocean_time=frc_time_days)
            q_sensible.name = 'Q_S'
            
            q_latent = ds_frc['latent'].mean(dim=['eta_rho', 'xi_rho'])
            q_latent = q_latent.assign_coords(ocean_time=frc_time_days)
            q_latent.name = 'Q_L'
            
            q_turb_net = q_sensible + q_latent
            turbulent_components_available = True
            
        elif 'shflux' in ds_frc.variables:
            # Only Net Turbulent Flux (shflux) is available
            q_turb_net = ds_frc['shflux'].mean(dim=['eta_rho', 'xi_rho'])
            q_turb_net = q_turb_net.assign_coords(ocean_time=frc_time_days)
            q_turb_net.name = 'Q_S + Q_L' # Rename for plotting
            
        # --- 4. Net Total Heat Flux (Q_H) ---
        # Q_H = Q_SW + Q_LW_net + Q_Turb_net
        q_net_total = q_sw + q_lw_net + q_turb_net
        
        # --- Plotting ---
        
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot the four main components (W/m^2, grid average)
        q_sw.plot(ax=ax, label='$Q_{SW}$ (Shortwave)', color='red', linestyle='-')
        q_lw_net.plot(ax=ax, label='$Q_{LW}^{\\text{net}}$ (Net Longwave)', color='purple', linestyle='--')
        
        if turbulent_components_available:
            # Plot separate sensible and latent fluxes
            q_sensible.plot(ax=ax, label='$Q_S$ (Sensible)', color='orange', linestyle='-')
            q_latent.plot(ax=ax, label='$Q_L$ (Latent)', color='darkgreen', linestyle='-')
            
        elif 'shflux' in ds_frc.variables:
            # Plot net turbulent flux
            q_turb_net.plot(ax=ax, label='$Q_S + Q_L$ (Net Turbulent)', color='green', linestyle='-.')
        else:
            # If no turbulent fluxes are found, plot zero line for clarity (if not zero, the Q_net will be wrong)
            ax.plot(frc_time_days.values, q_turb_net.values, label='$Q_S + Q_L$ (Assumed Zero)', color='gray', linestyle=':')


        # Plot the total net flux
        q_net_total.plot(ax=ax, label='$Q_H$ (Net Total Heat Flux)', color='black', linewidth=3.0)

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
        
        title_eq = '$Q_H = Q_{SW} + Q_{LW}^{\\text{net}} + Q_S + Q_L$'
        ax.set_title(f'Thermodynamic Heat Flux Components (Grid Average)\nEquation: {title_eq}')
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Heat Flux (W/m$^2$)')
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            print(f"Figure saved to {filename}")
        else:
            plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
    except KeyError as e:
        print(f"Error: A required variable is missing from a file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Debug: Error type was {type(e)}")

def calculate_model_heat_stats(ds_his, rho0=1025.0, Cp=3985.0):
    """
    Calculates total heat content change and the rate of change from ROMS history data.

    Parameters
    ----------
    ds_his : xarray dataset
        ROMS history data
    rho0 : real scalar default 1025.0  
        Reference density of seawater (kg/m^3)
    Cp : real scalar, default 3985.0 
        Specific heat of seawater (J/(kg*K))
    """
    temp = ds_his['temp']
    pm = ds_his['pm']
    pn = ds_his['pn']
    z_w = ds_his['z_w']
    
    # Time handling
    his_time_float_sec, his_time_days = robust_time_conversion(ds_his['ocean_time'])
    dt_his = np.mean(np.diff(his_time_float_sec))
    
    # Calculate area and volume
    try:
        area_scalar = 1.0 / (pm.isel(eta_rho=0, xi_rho=0).item() * pn.isel(eta_rho=0, xi_rho=0).item())
    except Exception:
        print("Advarsel: Grid-variablene pm/pn har uventede dimensjoner. Bruker standard 1x1 areal.")
        area_scalar = 1.0
        
    Hz = z_w.diff(dim='s_w').rename({'s_w': 's_rho'})
    volume_per_layer_aligned = xr.DataArray(
        Hz.values * area_scalar,
        coords=temp.coords,
        dims=temp.dims
    )

    # Calculate Total Heat Content (J)
    initial_temp = temp.isel(ocean_time=0, drop=True) 
    temp_anomaly = temp - initial_temp
    heat_content_per_cell = rho0 * Cp * temp_anomaly * volume_per_layer_aligned
    total_heat_content = heat_content_per_cell.sum(dim=['eta_rho', 'xi_rho', 's_rho'])
    total_heat_content = total_heat_content.assign_coords(ocean_time=his_time_days)

    # Calculate Rate of Change (W)
    rate_of_change_hc = np.diff(total_heat_content.values) / dt_his
    time_midpoints_days = (his_time_float_sec[:-1] + (dt_his / 2.0)) / SEC_PER_DAY #(24.0 * 3600.0)
    rate_of_change_da = xr.DataArray(rate_of_change_hc, coords=[('ocean_time', time_midpoints_days)])

    return total_heat_content, rate_of_change_da, area_scalar

def calculate_forcing_heat_stats(ds_frc, ds_his, area_scalar):
    """
    Calculates integrated forcing flux and cumulative heat input from forcing data.
    """
    frc_time_float_sec, frc_time_days = robust_time_conversion(ds_frc['ocean_time'])
    his_time_float_sec, his_time_days = robust_time_conversion(ds_his['ocean_time'])
    
    # Determine Net Heat Flux (including LW calculation if necessary)
    if 'shflux' in ds_frc.variables:
        net_heat_flux = ds_frc['shflux']
    elif 'lwrad_down' in ds_frc.variables and 'swrad' in ds_frc.variables:
        epsilon = 0.97  # Emissivity of sea surface
        sigma = 5.67e-8 # Stefan-Boltzmann constant (W m^-2 K^-4)

        sst_kelvin = ds_his['temp'].isel(s_rho=-1, drop=True) + 273.15
        dl_out = (epsilon * sigma * sst_kelvin**4).assign_coords(ocean_time=his_time_days)
        
        dl_out_interp = dl_out.interp(
            ocean_time=frc_time_days, 
            method='linear', 
            kwargs={"fill_value": "extrapolate"}
        )
        net_heat_flux = ds_frc['swrad'] + (ds_frc['lwrad_down'] - dl_out_interp)
    else:
        raise KeyError("No valid heat flux variable found in forcing file.")

    # Calculate Integrated Flux and Cumulative Input
    dt_frc = np.mean(np.diff(frc_time_float_sec)) if len(frc_time_float_sec) > 1 else (his_time_float_sec[1] - his_time_float_sec[0])
    
    forcing_flux_sum = (net_heat_flux * area_scalar).fillna(0.0).sum(dim=['eta_rho', 'xi_rho'])
    forcing_flux_sum = forcing_flux_sum.assign_coords(ocean_time=frc_time_days)
    
    cumulative_forcing_input = (forcing_flux_sum * dt_frc).cumsum(dim='ocean_time')

    return forcing_flux_sum, cumulative_forcing_input

def verify_heat_content(his_file: str, frc_file: str, filename: str=''):
    """
    Calculates the total heat content in a ROMS simulation and compares it
    with the cumulative heat flux input from a forcing file.
    """
    try:
        ds_his = xr.open_dataset(his_file)
        ds_frc = xr.open_dataset(frc_file)

        # 1. Calculate Model Side
        total_hc, rate_hc, area = calculate_model_heat_stats(ds_his)
        
        # 2. Calculate Forcing Side
        forcing_flux, cum_forcing = calculate_forcing_heat_stats(ds_frc, ds_his, area)

        # 3. Plotting
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))

        total_hc.plot(ax=axes[0], label='Model Heat Change', marker='o')
        cum_forcing.plot(ax=axes[0], label='Cumulative Forcing Input', marker='x', linestyle='--')
        axes[0].set_title('Comparison: Total Heat Content vs. Cumulative Flux')
        axes[0].legend()
        axes[0].grid(True)

        rate_hc.plot(ax=axes[1], label='Model Heat Rate (W)', marker='o')
        forcing_flux.plot(ax=axes[1], label='Total Forcing Flux (W)', marker='x', linestyle='--')
        axes[1].set_title('Comparison: Heat Change Rate')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        # plt.show()
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
    except KeyError as e:
        print(f"Error: A required variable is missing from a file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Debug: Error type was {type(e)}")

def compare_heat_content(folder1: str, folder2: str, filename: str=''):
    """
    Calculates the total heat content in a ROMS simulation and compares it
    with the cumulative heat flux input from a forcing file.
    """
    exp_num1 = extract_experiment_number(folder1)
    exp_num2 = extract_experiment_number(folder2)

    his_file1 = os.path.join(folder1, 'roms_his.nc')
    frc_file1 = os.path.join(folder1, 'roms_frc.nc')
    his_file2 = os.path.join(folder2, 'roms_his.nc')
    frc_file2 = os.path.join(folder2, 'roms_frc.nc')


    try:
        # 3. Plotting
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
       
        for num, his_file, frc_file in [(exp_num1, his_file1, frc_file1), 
                                        (exp_num2, his_file2, frc_file2)]:
            ds_his = xr.open_dataset(his_file)
            ds_frc = xr.open_dataset(frc_file)

            # 1. Calculate Model Side
            total_hc, rate_hc, area = calculate_model_heat_stats(ds_his)
            
            # 2. Calculate Forcing Side
            forcing_flux, cum_forcing = calculate_forcing_heat_stats(ds_frc, ds_his, area)
       

            total_hc.plot(ax=axes[0], label=f'Model Heat Change exp.nr={num}', marker='o')
            cum_forcing.plot(ax=axes[0], label=f'Cumulative Forcing Input exp.nr={num}', marker='x', linestyle='--')
            rate_hc.plot(ax=axes[1], label=f'Model Heat Rate (W) exp.nr={num}', marker='o')
            forcing_flux.plot(ax=axes[1], label=f'Total Forcing Flux (W) exp.nr={num}', marker='x', linestyle='--')

        axes[0].set_title('Comparison: Total Heat Content vs. Cumulative Flux')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].set_title('Comparison: Heat Change Rate')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        # plt.show()
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
    except KeyError as e:
        print(f"Error: A required variable is missing from a file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Debug: Error type was {type(e)}")

    
def plot_heat_flux_components(frc_file: str, his_file: str, filename=None):
    """
    Plots individual heat flux components and their sum.
    Radiative: From ds_frc
    Turbulent (Latent/Sensible): From ds_his
    """
    ds_frc = xr.open_dataset(frc_file)
    ds_his = xr.open_dataset(his_file)

    _, frc_days = robust_time_conversion(ds_frc['ocean_time'])
    _, his_days = robust_time_conversion(ds_his['ocean_time'])
  
    # 1. Radiative Fluxes (Forcing)
    sw = ds_frc['swrad'].mean(dim=['eta_rho', 'xi_rho'])
    
    # Net Longwave calculation
    if 'lwrad' in ds_frc:
        lw_net = ds_frc['lwrad'].mean(dim=['eta_rho', 'xi_rho'])
    elif 'lwrad_down' in ds_frc and ds_his is not None:
        epsilon, sigma = 0.97, 5.67e-8
        sst_k = ds_his['temp'].isel(s_rho=-1).mean(dim=['eta_rho', 'xi_rho']) + 273.15
        sst_k = ds_his['temp'].isel(s_rho=-1).mean(dim=['eta_rho', 'xi_rho']) + 273.15
        lw_up = (-(epsilon * sigma * sst_k**4)).assign_coords(ocean_time=his_days)
        lw_down = ds_frc['lwrad_down'].mean(dim=['eta_rho', 'xi_rho'])
        # Interpolate upgoing LW to match forcing time
        lw_net = lw_down + lw_up.interp(ocean_time=frc_days)
    else:
        lw_net = xr.zeros_like(sw)

    # 2. Turbulent Fluxes (History - using ROMS standard variable names)
    # latent heat flux is often 'lhflux' and sensible is 'shflux' in history files
    latname = 'latent' #'lhflux'
    senname = "sensible" # 'shflux'
    latent = ds_his[latname].mean(dim=['eta_rho', 'xi_rho']).assign_coords(ocean_time=his_days) if latname in ds_his else 0
    sensible = ds_his[senname].mean(dim=['eta_rho', 'xi_rho']).assign_coords(ocean_time=his_days) if senname in ds_his else 0

    # 3. Align Turbulent Fluxes to Forcing Time
    latent_interp = latent.interp(ocean_time=frc_days) if isinstance(latent, xr.DataArray) else 0
    sensible_interp = sensible.interp(ocean_time=frc_days) if isinstance(sensible, xr.DataArray) else 0
    
    # Calculate Net
    net_flux = sw + lw_net + latent_interp + sensible_interp

    # 4. Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(frc_days, sw, label='Shortwave (Into Water)', color='gold', alpha=0.8)
    plt.plot(frc_days, lw_net, label='Net Longwave', color='red', alpha=0.7)
    plt.plot(frc_days, latent_interp, label='Latent (From History)', color='blue', alpha=0.6)
    plt.plot(frc_days, sensible_interp, label='Sensible (From History)', color='green', alpha=0.6)
    
    # Bold Net Flux
    plt.plot(frc_days, net_flux, color='black', linewidth=2.5, label='NET HEAT FLUX')
    
    # Shade regions for clarity
    plt.fill_between(frc_days, net_flux, 0, where=(net_flux > 0), color='orange', alpha=0.2, label='Heating Ocean')
    plt.fill_between(frc_days, net_flux, 0, where=(net_flux < 0), color='cyan', alpha=0.2, label='Cooling Ocean')

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Surface Heat Flux Components (Positive = Heating Ocean)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Flux (W/m²)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    # plt.show()


def plot_conservative_temp(romsfile, filename=None):
    """Open history files and plot conservative temperature profiles"""
    f = Dataset(romsfile, 'r')

    # Get vertical coordinate
    z_rho = f.variables['z_rho'][:,:,7,6]

    # Get density 
    rho = f.variables['rho'][:,:,7,6]

    # Get salt
    salt = f.variables['salt'][:,:,7,6]

    # Get temp 
    temp = f.variables['temp'][:,:,7,6]

    # Nlevels = [0,1,2]
    # Ncolumns = [0,1,2]

    # z_rho = np.transpose(z_rho, (1,0,2,3)).reshape((Nlevels, Ncolumns))
    # salt = np.transpose(salt, (1,0,2,3)).reshape((Nlevels, Ncolumns))
    # temp = np.transpose(temp, (1,0,2,3)).reshape((Nlevels, Ncolumns))

    # # Remove columns on land and reverse layers so that surface comes first
    # not_nan_ind = np.where(~np.isnan(z_rho[0,:]))[0]
    # z_rho = z_rho[::-1, not_nan_ind]
    # salt = salt[::-1, not_nan_ind]
    # temp = temp[::-1, not_nan_ind]

    # Convert salinity and temperature from ROMS to TEOS-10 standards
    p = gsw.conversions.p_from_z(z_rho, lat=60.0)
    SA = gsw.conversions.SA_from_SP(salt, p, lon=0.0, lat=60.0)
    CT = gsw.conversions.CT_from_pt(SA, temp)


    # Get time
    otime = f.variables['ocean_time'][:]
    otime = otime/(24*3600.)
    ny = np.shape(rho)[1]
    dt = np.array([otime,]*ny).transpose()

    # Check if we should limit range
    # if maxdensity > 0.0:
    #     rho[rho>maxdensity] = np.nan

    # Open figure
    
    plt.figure()
    print(np.shape(dt))
    print(np.shape(CT))
    print(np.shape(z_rho))

    # Plot filled contours
    # plt.contourf(dt, z_rho, salt, 200, cmap=cm.ocean_r)
    # plt.contourf(dt, z_rho, temp, 200, cmap=cm.ocean_r)
    plt.contourf(dt, z_rho, CT, 200, cmap=cm.ocean_r)
    plt.colorbar(label=r'Conservative Temperature [$C^{\circ}$]')
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    if filename:
        plt.savefig(filename)
    # plt.show()

    # Close file
    f.close()

def plot_absolute_salinity(romsfile, filename=None):
    """Open history files and plot absolute salinity profiles"""
    f = Dataset(romsfile, 'r')

    # Get vertical coordinate
    z_rho = f.variables['z_rho'][:,:,7,6]

    # Get density 
    rho = f.variables['rho'][:,:,7,6]

    # Get salt
    salt = f.variables['salt'][:,:,7,6]

    # Get temp 
    temp = f.variables['temp'][:,:,7,6]

    # Convert salinity and temperature from ROMS to TEOS-10 standards
    p = gsw.conversions.p_from_z(z_rho, lat=60.0)
    SA = gsw.conversions.SA_from_SP(salt, p, lon=0.0, lat=60.0)


    # Get time
    otime = f.variables['ocean_time'][:]
    otime = otime/(24*3600.)
    ny = np.shape(rho)[1]
    dt = np.array([otime,]*ny).transpose()

    # Check if we should limit range
    # if maxdensity > 0.0:
    #     rho[rho>maxdensity] = np.nan

    # Open figure
    
    plt.figure()
    print(np.shape(dt))
    print(np.shape(SA))
    print(np.shape(z_rho))

    # Plot filled contours
    # plt.contourf(dt, z_rho, salt, 200, cmap=cm.ocean_r)
    # plt.contourf(dt, z_rho, temp, 200, cmap=cm.ocean_r)
    plt.contourf(dt, z_rho, SA, 200, cmap=cm.ocean_r)
    plt.colorbar(label=r'Absolute Salinity [$g/kg$]')
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    if filename:
        plt.savefig(filename)
    # plt.show()

    # Close file
    f.close()



def main(exp_name, useflux=True, diurnal=True):
    """ 
    - Makes flux forcing file or bulk forcing file
    - Run current compiled ROMS model using the flux forcing file 
      and saves the results to the RESULT_FOLDER.
      The build roms script is also copied to the result folder
    - Plot hovmuller density profile and save the figure on pdf format
    """
    if useflux:
        make_flux_force_file()
    else:
        make_bulkforce_file(diurnal)
    folder = os.path.join(RESULT_FOLDER, now() + exp_name)
    run_roms(folder)
    ext = '.png'
    romsfile = os.path.join(folder, 'roms_his.nc')
    forcing_file = os.path.join(folder, 'roms_frc.nc')
    plot_density_hovmuller(romsfile,
                           maxdensity=0.0,
                           filename=os.path.join(folder, 'density_hovmuller' +ext)
                           )
    plot_speed_hovmuller(romsfile,
                         filename=os.path.join(folder, 'speed_hovmuller' +ext)
                         )

    plot_hodograph([romsfile], savefile=True)
    plot_conservative_temp(romsfile, filename=os.path.join(folder, 'conservative_temp' +ext))
    plot_absolute_salinity(romsfile, filename=os.path.join(folder, 'absolute_salinity' +ext))
    if useflux:
        verify_heat_content(romsfile, forcing_file, filename=os.path.join(folder, 'verify_heat_content' + ext))
     
    plt.show()


def find_latest_run_folder(exp_name: str) -> str:
    """Finds the path to the most recently created run folder with a given experiment name."""
    list_of_folders = glob.glob(os.path.join(RESULT_FOLDER, f'*-*{exp_name}*'))
    if not list_of_folders:
        raise FileNotFoundError(f"Found no foldere with '{exp_name}' in {RESULT_FOLDER}")
    latest_folder = max(list_of_folders, key=os.path.getctime)
    base_folder = os.path.basename(latest_folder)
    return base_folder


def plots():
    # for folder in ['2025-03-09T164757no_F_cooling_0_xstress_0p5']: 
    folders = [find_latest_run_folder('_exp15')]
    #folders = os.listdir(RESULT_FOLDER)
    for folder in folders:
        if True: #folder.startswith('2025-09-24'):# 
            print(folder)
            root = os.path.join(RESULT_FOLDER, folder)
            romsfile = os.path.join(root, 'roms_his.nc')
            forcing_file = os.path.join(root, 'roms_frc.nc')
            ext = '.png'
            #plot_hodograph([romsfile], savefile=True)
            #plot_density_hovmuller(romsfile, maxdensity=0.0, filename=os.path.join(root, 'density_hovmuller' + ext))
            #plot_speed_hovmuller(romsfile, filename=os.path.join(root, 'speed_hovmuller' +ext))
            #plot_conservative_temp(romsfile, filename=os.path.join(root, 'conservative_temp' +ext))
            #plot_absolute_salinity(romsfile, filename=os.path.join(root, 'absolute_salinity' +ext))
            #verify_heat_content(romsfile, forcing_file, filename=os.path.join(root, 'verify_heat_content' + ext))
            plot_thermodynamic_fluxes(romsfile, forcing_file, filename=os.path.join(root, 'thermodynamic_fluxes' + ext))
            #plt.show()
            #calculate_roms_cell_volume(romsfile)


def plot_kaihc(ext='.png'):
    kaifolder = find_latest_run_folder('_kaihc')
    print(kaifolder)
    root = os.path.join(RESULT_FOLDER, kaifolder)
    for experiment in os.listdir(root):
        if experiment.startswith('U10_'):
            folder = os.path.join(root, experiment)
            romsfile = os.path.join(folder,  f'KHC-his.nc-{experiment}')
            forcing_file = os.path.join(folder, f'roms_bulkforce.nc-{experiment}')
            plot_density_hovmuller(romsfile,
                           maxdensity=0.0,
                           filename=os.path.join(folder, 'density_hovmuller' +ext)
                           )
            plot_speed_hovmuller(romsfile,
                         filename=os.path.join(folder, 'speed_hovmuller' +ext)
                         )

            plot_hodograph([romsfile], savefile=True)
            plot_conservative_temp(romsfile, filename=os.path.join(folder, 'conservative_temp' +ext))
            plot_absolute_salinity(romsfile, filename=os.path.join(folder, 'absolute_salinity' +ext))



if __name__ == "__main__":
    #main(exp_name='_exp30_Ninfo_1_strat_F_swradmax_800_bulk_Uwind_5_cloud_0_Qair_80_Tair_20_Pair_1020',
    #     useflux=False, diurnal=True)
    #folder=os.path.join(RESULT_FOLDER, '2026-01-21T081848_exp24_Ninfo_1_strat_F_swrad_300_bulk_Uwind_5_cloud_0_Qair_80_Tair_10_Pair_1020')
    folder = os.path.join(RESULT_FOLDER, find_latest_run_folder('_exp30'))
    history_file = os.path.join(folder, 'roms_his.nc')
    forcing_file = os.path.join(folder, 'roms_frc.nc')
    # folder=os.path.join(RESULT_FOLDER, '2025-12-18_kaihc','U10_5-cloud_0-swrad_300')
    # history_file = os.path.join(folder, 'KHC-his.nc-U10_5-cloud_0-swrad_300')
    # forcing_file = os.path.join(folder, 'roms_bulkforce.nc-U10_5-cloud_0-swrad_300')

    # verify_heat_content(history_file, forcing_file, filename=os.path.join(folder, 'verify_heat_content.png'))
    plot_heat_flux_components(forcing_file, history_file, filename=os.path.join(folder, 'heat_flux_components.png'))

    # exp1, exp2 = ('_exp28', '_exp29')
    # folder1 = os.path.join(RESULT_FOLDER, find_latest_run_folder(exp1))
    # folder2 = os.path.join(RESULT_FOLDER, find_latest_run_folder(exp2))
    # compare_heat_content(folder1, folder2, filename=os.path.join(folder2, f'compare_heat_content{exp1}{exp2}.png'))

    #plots()

    #print_forcing_info(os.path.join(RESULT_FOLDER, 
    #                                '2025-03-30T164056_exp6_strat_no_F_cooling_m100_xstress_0p1',
    #                                'roms_frc.nc'))
    #plot_kaihc()