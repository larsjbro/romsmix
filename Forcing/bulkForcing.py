#!/usr/bin/python

from netCDF4 import Dataset
import numpy as np
import os

# Generate empty forcing file and open for editing
os.system('ncgen -b roms_frc.cdl')
f = Dataset('roms_frc.nc', 'a')

# Make list of forcing variables to set (must match "roms_frc.cdl"!)
frcvar = ['swrad', 'shflux', 'swflux', 'sustr', 'svstr']

# Loop over variables and initialize to zero.
#
# NOTE! For more advanced input, edit yourself.
#
for varname in frcvar:
    print varname
    var = f.variables[varname]
    var[:] = 0.0

# Setting constant stress in x-direction
sustr = f.variables['sustr']
sustr[:] = 0.2

# Set time variable, the end time is the important one.
#
# NOTE! If time varying input, which requires more than two
# time steps, remember to edit the size of "ocean_time" at
# the start of "roms_frc.cdl".
#
ot = f.variables['ocean_time']
ot[:] = [0.0,48*3600.0] 

# Sync file to force write, then close.
f.sync()
f.close()
