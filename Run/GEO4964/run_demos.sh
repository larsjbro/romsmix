#!/bin/bash

# All cases are one week simulations starting from rest with initial linear 
# temperature stratification (salinity is constant). Note that we use a 
# linear EOS, and that the initial temperature profile is such that 
#
# dT/dz = 1/20 C/m
#
# which implies that the buoyancy frequency is 
#
# N = 0.0091 s^-1, 
#
# hence the period is 2*pi/N = 688 s, or about 11.5 minutes. 
#
# The Richardson number Ri = S^2/N^2 becomes critical for Ri < 0.25, 
# which in our case is the case when the vertical shear exceeds 
#
# S = 0.0046 s^-1. 
#
# With constant stress 0.2 N/m^2, and a constant viscosity nu = 1.0e-3 m^2/s, 
# we have du/dz = 0.19 s^-1 at the surface from start. Note that in
# ROMS the acc. of gravity is constant 9.81 m/s^2, see mod_scalars.
#
# NOTE: to run this demo you need to compile the various ROMS executables
# with the relevant functionality and rename them so that they match 
# the names below.

set -x 

# We start by running a few simulations with no net heat flux and constant
# wind stress in the x-direction:

cp roms_frc.nc-constant_wind_xdir roms_frc.nc

# First case is with constant viscosity/diffusivities, with and without 
# Coriolis parameter (off or equiv. to 60N):

./romsS_NOMIX_NO_F < ../../External/roms_column.in
mv roms_his.nc roms_his.nc-nomix_no_f
./romsS_NOMIX_F < ../../External/roms_column.in
mv roms_his.nc roms_his.nc-nomix_f

# Second case is with KPP, with and without 
# Coriolis parameter (off or equiv. to 60N):

./romsS_KPP_NO_F < ../../External/roms_column.in
mv roms_his.nc roms_his.nc-kpp_no_f
./romsS_KPP_F < ../../External/roms_column.in
mv roms_his.nc roms_his.nc-kpp_f

# Third case is for comparison between KPP and GLS, with Coriolis (60N):

./romsS_GLS_F < ../../External/roms_column.in
mv roms_his.nc roms_his.nc-gls_f

# Finally, fourth case is a comparison between GLS and KPP 
# for a case with constant, rather weak winds (tau = 0.05 N/m^2)
# and constant cooling at a rate of -100 W/m^2:

cp roms_frc.nc-weak_wind_and_cooling roms_frc.nc
./romsS_KPP_F < ../../External/roms_column.in
mv roms_his.nc roms_his.nc-kpp_cooling
./romsS_GLS_F < ../../External/roms_column.in
mv roms_his.nc roms_his.nc-gls_cooling





