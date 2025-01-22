from netCDF4 import Dataset
import numpy as np
import os

# Read input file
f_in = Dataset('atmos_20190723-20190731.nc', 'r')
i_time = f_in.variables['time']


# Generate empty forcing file and open for editing
os.system('ncgen -b roms_bulkforce.cdl')
f = Dataset('roms_bulkforce.nc', 'a')

# Make list of forcing variables to set (must match "roms_bulkfrc.cdl"!)
frcvar = ['cloud', 'Uwind', 'Vwind', 'Pair', 'Tair', 'Qair', 'rain']

# Loop over variables and initialize to zero.
#
# NOTE! For more advanced input, edit yourself.
#
for varname in frcvar:
    print(varname)
    var = f.variables[varname]
    varin = f_in.variables[varname]
    var[:,:,:] = varin[:]

# Set time variable, the end time is the important one
ot = f.variables['ocean_time']
ot[:] = i_time[:]


# Sync file to force write, then close.
f.sync()
f.close()
f_in.close()
