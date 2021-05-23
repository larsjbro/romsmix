#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import argparse

# Get command line arguments
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot surface current hodograph.')
    parser.add_argument('romsfile', type=str, help='name of ROMS history file')
    args = parser.parse_args()

# Open history files and plot profiles
f = Dataset(args.romsfile, 'r')

# Get velocities at rho points
u = f.variables['u_eastward'][:,-1,7,6]
v = f.variables['v_northward'][:,-1,7,6]

# Plot hodograph
plt.figure()

dt = 3600 # Output time step
plt.plot([0],[0],'rd')
plt.plot(3600*u, 3600*v)
plt.axis('equal')
plt.show()

# Close file
f.close()
