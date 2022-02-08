import numpy as np
import argparse
from matplotlib import pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset

# Get command line arguments
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot TKE Hovmuller diagram.')
    parser.add_argument('romsfile', type=str, help='name of ROMS history file')
    args = parser.parse_args()


# Open history files and plot profiles
f = Dataset(args.romsfile, 'r')

# Get vertical coordinate
z_w = f.variables['z_w'][:,:,7,6]

# Get TKE
akv = f.variables['AKv'][:,:,7,6]

# Get time
otime = f.variables['ocean_time'][:]
otime = otime/(24*3600.)
dt = np.array([otime,]*401).transpose()

# Open figure
plt.figure()

# Plot filled contours
plt.contourf(dt,z_w,np.log(akv),vmax=-1.5,vmin=-12)

# Add info 
plt.colorbar(label='Eddy viscosity [m^2/s]')
plt.xlabel('Days')
plt.ylabel('Depth [m]')
plt.show()

# Close file
f.close()
