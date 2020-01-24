import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset

# Open history file
f = Dataset('../Run/roms_his_rot.nc', 'r')
f2 = Dataset('../Run/roms_his.nc', 'r')

# Get vertical coordinate
z_r = f.variables['z_rho'][:,:,7,6]

# Get density profile
rho = f.variables['rho'][:,:,7,6]
rho2 = f2.variables['rho'][:,:,7,6]

# Plot density profiles
plt.figure()

# All of them
for i in range(z_r.shape[0]):
    plt.plot(rho[i,:], z_r[i,:])

# First and last only
#plt.plot(rho[0,:], z_r[0,:], 'b-')
#plt.plot(rho[-1,:], z_r[-1,:], 'r-')

plt.show()

# Plot density profiles
plt.figure()

# All of them
for i in range(z_r.shape[0]):
    plt.plot(rho2[i,:], z_r[i,:])

# First and last only
#plt.plot(rho[0,:], z_r[0,:], 'b-')
#plt.plot(rho[-1,:], z_r[-1,:], 'r-')

plt.show()

# Close file
f.close()
