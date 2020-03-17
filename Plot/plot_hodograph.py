import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset

# Open history file
f = Dataset('../Run/roms_his.nc', 'r')

# Get velocities at rho points
u = f.variables['u_eastward'][:,-1,7,6]
v = f.variables['v_northward'][:,-1,7,6]

# Plot hodograph
plt.figure()

plt.plot(u, v)
plt.axis('equal')
plt.show()

# Close file
f.close()
