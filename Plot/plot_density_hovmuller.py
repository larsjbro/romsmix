import numpy as np
import argparse
from matplotlib import pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset

# Get command line arguments
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot density Hovmuller diagram.')
    parser.add_argument('romsfile', type=str, help='name of ROMS history file')
    parser.add_argument('--maxdensity', type=float, default=0.0, help='cutoff density value to plot')
    args = parser.parse_args()


# Open history files and plot profiles
f = Dataset(args.romsfile, 'r')

# Get vertical coordinate
z_r = f.variables['z_rho'][:,:,7,6]

# Get density 
rho = f.variables['rho'][:,:,7,6]

# Get time
otime = f.variables['ocean_time'][:]
otime = otime/(24*3600.)
dt = np.array([otime,]*400).transpose()

# Check if we should limit range
if args.maxdensity > 0.0:
    rho[rho>args.maxdensity] = np.nan

# Open figure
plt.figure()

# Plot filled contours
plt.contourf(dt, z_r, rho, 200, cmap=cm.ocean_r)

# Add info 
plt.colorbar(label='Potential density anomaly [kg/m^3]')
plt.xlabel('Days')
plt.ylabel('Depth [m]')
plt.show(block=False)

# Close file
f.close()
