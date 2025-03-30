import os
import subprocess
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from netCDF4 import Dataset
FORCING_FOLDER = os.path.dirname(__file__)
ROOT = os.path.dirname(FORCING_FOLDER)
RUN_FOLDER = os.path.join(ROOT, 'Run')
EXTERNAL_FOLDER = os.path.join(ROOT, 'External')
INCLUDE_FOLDER = os.path.join(ROOT, 'Include')
RESULT_FOLDER = os.path.join(ROOT, 'Results')


def now():
    """Returns current date time on isoformat: yyyy-mm-ddThhmmss"""
    return datetime.now().isoformat("T", "seconds").replace(':', '')


def make_forcing_file():
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

    # Set time variable, the end time is the important one.
    #
    # NOTE! If time varying input, which requires more than two
    # time steps, remember to edit the size of "ocean_time" at
    # the start of "roms_frc.cdl".
    #
    ot = f.variables['ocean_time']
    #ot[:] = [0.0,7*24*3600.0] 
    num_time_steps = 169
    timevec = np.linspace(0.0, 7*24*3600.0, num_time_steps) 
    ot[:] = timevec

    # Setting shortwave diurnal cycle
    swrad = np.zeros_like(f.variables['swrad'][:])
    onedayfreq = 2*np.pi/(3600.*24)
    for i in range(num_time_steps):
        swrad[i,:,:] = -0.0*np.cos(onedayfreq*timevec[i])*np.ones((14,12))
        
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
    print("Finished generating roms_flux.nc", txt)

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


def plot_hodograph(romsfile, savefile=False):
    # Check how many input files
    N = len(romsfile)

    # Plot hodograph
    plt.figure()

    for i in range(N):

        f = []

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
            plt.savefig(os.path.join(folder, 'hodograph.pdf'))
        # plt.show()

        # Close file
        f.close()




def main(exp_name):
    """ 
    - Makes flux forcing file
    - Run current compiled ROMS model using the flux forcing file 
      and saves the results to the RESULT_FOLDER.
      The build roms script is also copied to the result folder
    - Plot hovmuller density profile and save the figure on pdf format
    """
    make_forcing_file()
    folder = os.path.join(RESULT_FOLDER, now() + exp_name)
    run_roms(folder)
    filename = os.path.join(folder, 'roms_his.nc')
    plot_density_hovmuller(filename,
                           maxdensity=0.0,
                           filename=os.path.join(folder, 'density_hovmuller.pdf')
                           )
    plot_speed_hovmuller(filename,
                         filename=os.path.join(folder, 'speed_hovmuller.pdf')
                         )

    plot_hodograph([filename], savefile=True)
    plt.show()


def plots():
    # for folder in ['2025-03-09T164757no_F_cooling_0_xstress_0p5']: 
    for folder in os.listdir(RESULT_FOLDER):
        if folder.startswith('2025-03-28'):
            print(folder)
            root = os.path.join(RESULT_FOLDER, folder)
            filename = os.path.join(root, 'roms_his.nc')
            plot_hodograph([filename], savefile=True)
            # plot_density_hovmuller(filename, maxdensity=0.0, filename=os.path.join(root, 'density_hovmuller.pdf'))
            plot_speed_hovmuller(filename, filename=os.path.join(root, 'speed_hovmuller.pdf'))
            plt.show()


if __name__ == "__main__":
    main(exp_name='_exp3_2Xstrat_no_F_cooling_0_xstress_0p1')
    #plots()
