#!/usr/bin/python3

import numpy as np
import argparse
from matplotlib import pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset


def plot_density_hovmuller(romsfile, maxdensity):
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
    plt.show()

    # Close file
    f.close()


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser(description='Plot density Hovmuller diagram.')
    parser.add_argument('romsfile', type=str, help='name of ROMS history file')
    parser.add_argument('--maxdensity', type=float, default=0.0, help='cutoff density value to plot')
    args = parser.parse_args()
    plot_density_hovmuller(args.romsfile, args.maxdensity)


if __name__ == "__main__":
    #main()
    #plot_profiles('../Run/roms_his.nc', maxdensity=0.0)
    print('hei')
    plot_density_hovmuller('../Run/roms_his.nc', maxdensity=0.0)
    
