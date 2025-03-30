from netCDF4 import Dataset
import numpy as np
import os

# Generate empty forcing file and open for editing
os.system('ncgen -b roms_flux.cdl')
f = Dataset('roms_flux.nc', 'a')

# Make list of forcing variables to set (must match "roms_frc.cdl"!)
frcvar = ['swrad', 'shflux', 'swflux', 'sustr', 'svstr']

# Loop over variables and initialize to zero.
#
# NOTE! For more advanced input, edit yourself.
#
for varname in frcvar:
    print(varname)
    var = f.variables[varname]
    var[:] = 0.0

# Setting stress in x-direction
sustr = f.variables['sustr']
sustr[:] = 0.0
# sustr[:] = 0.05

# Setting constant cooling
shflux = f.variables['shflux']
shflux[:] = 0.0
#shflux[2:8] = -100.0

# Set time variable, the end time is the important one.
#
# NOTE! If time varying input, which requires more than two
# time steps, remember to edit the size of "ocean_time" at
# the start of "roms_frc.cdl".
#
ot = f.variables['ocean_time']
#ot[:] = [0.0,7*24*3600.0] 
num_time_steps = 169
timevec = np.linspace(0.0,7*24*3600.0,num_time_steps) 
ot[:] = timevec

# Setting shortwave diurnal cycle
swrad = np.zeros_like(f.variables['swrad'][:])
onedayfreq = 2*np.pi/(3600.*24)
for i in range(num_time_steps):
    swrad[i,:,:] = -0.0*np.cos(onedayfreq*timevec[i])*np.ones((14,12))
    
swrad[np.where(swrad<0)] = 0.0

f.variables['swrad'][:,:,:] = swrad

# Experiment with constant magnitude wind stress for rotating case, but with
# constant direction in an absolute sense. Assuming Coriolis parameter is
# consistent with 60 deg N.
sustr = np.zeros_like(f.variables['sustr'])
svstr = np.zeros_like(f.variables['svstr'])
f0 = 0.00012630313674635122
#for i in range(169):
#    sustr[i,:,:] = 0.2*np.cos(f0*timevec[i]*np.ones((14,11)))
#    svstr[i,:,:] = -0.2*np.sin(f0*timevec[i]*np.ones((13,12)))

f.variables['sustr'][:,:,:] = sustr
f.variables['svstr'][:,:,:] = svstr

# Sync file to force write, then close.
f.sync()
f.close()

# Rename file
os.system('mv roms_flux.nc roms_frc.nc')

