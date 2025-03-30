#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import argparse


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser(description='Plot surface current hodograph.')
    parser.add_argument('romsfile', type=str, nargs='+', help='name of ROMS history file(s)')
    args = parser.parse_args()
    plot_hodograph(args.romsfile)


def plot_hodograph(romsfile):
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
        plt.show()

        # Close file
        f.close()


if __name__ == "__main__":
    # main()
    plot_hodograph(['../Run/roms_his.nc'])