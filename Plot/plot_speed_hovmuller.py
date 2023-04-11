#!/home/kaihc/miniconda3/envs/roms/bin/python
import numpy as np
import argparse
from matplotlib import pyplot as plt
from matplotlib import colors
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
z_r = f.variables['z_rho'][:,:,7,6]

# Get u,v
u = f.variables['u'][:,:,7,6]
v = f.variables['v'][:,:,7,6]
speed = np.sqrt(u**2 + v**2)
speed[speed>0.2] = 0.2

# Get time
otime = f.variables['ocean_time'][:]
otime = otime/(24*3600.)
dt = np.array([otime,]*42).transpose()

# Open figure
plt.figure()

# Plot filled contours
#plt.contourf(dt,z_r,speed,levels=np.logspace(-6,0,25),norm=colors.LogNorm())
plt.contourf(dt,z_r,speed,25)

# Add info 
plt.colorbar(label='Speed [m/s]')
plt.xlabel('Days')
plt.ylabel('Depth [m]')
plt.show()

# Close file
f.close()
