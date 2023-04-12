#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import argparse

# Get command line arguments
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot surface current hodograph.')
    parser.add_argument('romsfile', type=str, nargs='+', help='name of ROMS history file(s)')
    args = parser.parse_args()

# Check how many input files
N = len(args.romsfile)

# Plot hodograph
plt.figure()

for i in range(N):

    f = []

    # Open history files and plot profiles
    f = Dataset(args.romsfile[i], 'r')

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
    plt.show(block=False)

    # Close file
    f.close()
