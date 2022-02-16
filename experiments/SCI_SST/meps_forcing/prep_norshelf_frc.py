# Execute these commands before running script to load dependencies:

# module load fou-hi/pyromstools/dev
# source /modules/centos7/conda/Feb2021/etc/profile.d/conda.sh
# conda activate production-04-2021


import pyromstools
import os
import datetime

start_date = datetime.datetime(2021, 5, 1)
end_date = datetime.datetime(2021, 5, 10)

cfg_file = 'meps_det.cfg'
ncml_file = 'meps_det.ncml'

atm = pyromstools.MepsForcing(start_date, end_date, cfg_file, os.getcwd() )
atm.cfg.addattr("ncml", "config", ncml_file)
output_file = os.path.join(os.getcwd(), 'atm_frc.nc')
atm.get_forcing(output_file)
