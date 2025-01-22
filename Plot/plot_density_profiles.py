import numpy as np
import argparse
from matplotlib import pyplot as plt
from netCDF4 import Dataset

# Get command line arguments
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot density profiles.')
    parser.add_argument('romsfile', type=str, nargs='+', help='name of ROMS history file(s)')
    args = parser.parse_args()

# Check how many input files
N = len(args.romsfile)

# Plot density profiles
plt.figure()

# Open history files and plot profiles
f = []
for i in range(N):

    filename = args.romsfile[i]
    f.append(Dataset(filename, 'r'))

    # Get vertical coordinate
    z_r = f[i].variables['z_rho'][:,:,7,6]

    # Get density profile
    rho = f[i].variables['rho'][:,:,7,6]

    # Plot first and last only
    plt.plot(rho[0,:], z_r[0,:], linestyle='--', label=filename+' init')
    plt.plot(rho[-1,:], z_r[-1,:], linestyle='-', label=filename+' final')

# Add info 
plt.xlabel('Potential density anomaly [kg/m^3]')
plt.ylabel('Depth [m]')
plt.legend()
plt.show(block=False)

# Close files
for i in range(N):
    f[i].close()
