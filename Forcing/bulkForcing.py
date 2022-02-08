#!/usr/bin/python3
from netCDF4 import Dataset
import numpy as np
import os

# Generate empty forcing file and open for editing
os.system('ncgen -b roms_bulkforce.cdl')
f = Dataset('roms_bulkforce.nc', 'a')

# Make list of forcing variables to set (must match "roms_bulkfrc.cdl"!)
frcvar = ['cloud', 'Uwind', 'Vwind', 'Pair', 'Tair', 'Qair', 'rain', 'swrad']

# Loop over variables and initialize to zero.
#
# NOTE! For more advanced input, edit yourself.
#
for varname in frcvar:
    print(varname)
    var = f.variables[varname]
    var[:] = 0.0

# Setting constant wind in x-direction
f.variables['Uwind'][:,:,:] = 10.0

# Setting constant humidity
f.variables['Qair'][:,:,:] = 80.0

# Setting constant temperature
f.variables['Tair'][:,:,:] = 10.0

# Setting constant air pressure
f.variables['Pair'][:,:,:] = 1020.0

# Setting constant air pressure
f.variables['Pair'][:,:,:] = 1020.0

# Set time variable, the end time is the important one.
#
# NOTE! If time varying input, which requires more than two
# time steps, remember to edit the size of "ocean_time" at
# the start of "roms_frc.cdl".
#
ot = f.variables['ocean_time']
timevec = np.linspace(0.0,7*24*3600.0,169) 
ot[:] = timevec

# Setting shortwave diurnal cycle
swrad = np.zeros_like(f.variables['swrad'][:])
onedayfreq = 2*np.pi/(3600.*24)
for i in range(169):
    swrad[i,:,:] = -300.0*np.cos(onedayfreq*timevec[i])*np.ones((14,12))
    
swrad[np.where(swrad<0)] = 0.0

f.variables['swrad'][:,:,:] = swrad

# Sync file to force write, then close.
f.sync()
f.close()
