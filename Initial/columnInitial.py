#!/usr/bin/python

from netCDF4 import Dataset
import numpy as np
import os

# Copy restart file and open for editing
os.system('cp ini_blank.nc roms_ini.nc')
f = Dataset('roms_ini.nc', 'a')

# Make sure the time stamp is zero
ot = f.variables['ocean_time']
ot[:] = 0.0

# Initialize hydrography
salt = f.variables['salt']
for i in range(salt.shape[2]):
    for j in range(salt.shape[3]):
        salt[0,:,i,j] = np.linspace(35, 25, salt.shape[1])

temp = f.variables['temp']
for i in range(temp.shape[2]):
    for j in range(temp.shape[3]):
        temp[0,:,i,j] = np.linspace(5, 15, temp.shape[1])

# Sync file to force write, then close.
f.sync()
f.close()
