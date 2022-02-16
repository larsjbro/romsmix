import os, sys
import argparse
from traceback import format_exc
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from netCDF4 import Dataset

class Common:
    def __init__(self):
        print('\n\tLoading Common functions')

    def parse_args(self, raw_args):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--forcingType', dest = 'frcType' ,required = False,  default = 'bulk', help = 'bulk, flux, or ini')
        parser.add_argument('--inputfile', dest = 'inputFile', required = True, help = 'filepath to a ROMS forcing file (bulk or flux) or clm/his/ini')
        parser.add_argument('--starttime', dest = 'starttime', required = True, help ='start of the period to generate forcing for (YYYY-MM-DD)')
        parser.add_argument('--endtime', dest = 'endtime', required = True, help ='end of the period to generate forcing for (YYYY-MM-DD)')
        parser.add_argument('--latitude', dest =  'latitude', required = False, default = np.nan, type = float, help = 'latitude of location to generate forcing for')
        parser.add_argument('--longitude', dest =  'longitude', required = False, default = np.nan, type = float,  help = 'longitude of the location to generate forcing for')
        parser.add_argument('--outputfile', dest = 'outputFile', required = False, default = "roms_{}_frc_1D.nc", help = 'name of outputfile')
        parser.add_argument('--gridfile', dest = 'gridFile', required = False, default = None, help = 'gridfile with lon/lat information, in case forcing file does not hold this information')
        parser.add_argument('--Xgrid', dest = 'xgrid', required = False, default = np.nan, type = int, help = 'x index of point of interest')
        parser.add_argument('--Ygrid', dest = 'ygrid', required = False, default = np.nan, type = int, help = 'y index of point of interest')


        if raw_args:
            res = parser.parse_args(raw_args)
        else:
            res = parser.parse_args(sys.argv[1:])

        return res.frcType, res.inputFile, res.starttime, res.endtime, res.latitude, res.longitude, res.outputFile, res.gridFile, res.xgrid, res.ygrid

    def loadInput(self,ds):
        timedim, xdim ,ydim = self.dimensions(ds)

        # Reduce file to chosen location:
        ds = ds.isel({xdim : int(np.round(self.Xgrid)), ydim : int(np.round(self.Ygrid)) })
        # But if ROMS file with staggered grid, we also need to reduce u and v grid points...
        if  xdim.replace('rho', 'u') in ds.dims.keys():
            ds = ds.isel({xdim.replace('rho', 'u') : int(np.round(self.Xgrid)) + 1, ydim.replace('rho', 'u') : int(np.round(self.Ygrid))  })


        if  xdim.replace('rho', 'v') in ds.dims.keys():
            ds = ds.isel({xdim.replace('rho', 'v') : int(np.round(self.Xgrid)) , ydim.replace('rho', 'v') : int(np.round(self.Ygrid))  + 1 })



        # Reduce file to chosen period:
        if np.datetime64(self.endtime) < ds[timedim].values[0] or np.datetime64(self.starttime) > ds[timedim].values[-1]:
            print('\tERROR: Chosen time period ({} - {}) is outside scope of input file ({} - {})'.format(self.starttime, self.endtime, frc[timedim].values[0], frc[timedim].values[-1]))
            exit()
        if np.datetime64(self.starttime) < ds[timedim].values[0]:
            print('\tWARNING: Chosen period exceed scope of input file!')
            print('\tChosen: {} - {}'.format(self.starttime, self.endtime))
            print('\tOn file: {} - {}'.format(np.datetime_as_string(ds[timedim].values[0], unit='D'),np.datetime_as_string( ds[timedim].values[-1], unit='D')))
        if np.datetime64(self.endtime) > ds[timedim].values[-1]:
            print('\tWARNING: Chosen period exceed scope of input file!')
            print('\tChosen: {} - {}'.format(self.starttime, self.endtime))
            print('\tOn file: {} - {}'.format(np.datetime_as_string(ds[timedim].values[0], unit='D'),np.datetime_as_string( ds[timedim].values[-1], unit='D')))

        ds = ds.sel({timedim : slice(self.starttime, self.endtime) })
        return ds, ds.dims[timedim]

    def findIndex(self):
        coordinate_sets =  [['lon_rho', 'lat_rho'], ['lon', 'lat'], ['longitude', 'latitude']]
        has_coord = False
        for romsfile in [self.inputfile, self.gridfile]:
            if romsfile:
                grid = xr.open_dataset(romsfile)
                if any( [ coords[0] in grid.keys() and coords[1] in grid.keys() for coords in coordinate_sets ]):
                    has_coord = True
                    break

        if not has_coord:
            print('\tERROR: Unable to find lon/lat on input forcing- and/or grid-files!')
            print('\tTried variables {} on files {},{}'.format(coordinate_sets, self.inputfile, self.gridfile))
            print('\tExiting!!!')
            exit()


        for coordinates in coordinate_sets:
            try:
                x   = np.linspace(0, grid[coordinates[1]].values.shape[1]-1, grid[coordinates[1]].values.shape[1])
                y   = np.linspace(0, grid[coordinates[1]].values.shape[0]-1, grid[coordinates[1]].values.shape[0])
                xi  = np.zeros_like(grid[coordinates[1]].values)
                yi  = np.zeros([grid[coordinates[1]].values.shape[1], grid[coordinates[1]].values.shape[0]])
                xi[:,:] = x
                yi[:,:] = y
                yi  = np.swapaxes(yi, 1, 0)
                self.Xgrid = griddata( (grid[coordinates[0]].values.flatten(), grid[coordinates[1]].values.flatten()), xi.flatten(), (self.longitude, self.latitude) )
                self.Ygrid  = griddata( (grid[coordinates[0]].values.flatten(), grid[coordinates[1]].values.flatten()), yi.flatten(), (self.longitude, self.latitude) )
                break

            except:
                try:
                    gridlons, gridlats = np.meshgrid(grid[coordinates[0]].values, grid[coordinates[1]].values)
                    x   = np.linspace(0, gridlons.shape[1]-1, gridlons.shape[1])
                    y   = np.linspace(0, gridlons.shape[0]-1, gridlons.shape[0])
                    xi  = np.zeros_like(gridlats)
                    yi  = np.zeros([gridlons.shape[1], gridlons.shape[0]])
                    xi[:,:] = x
                    yi[:,:] = y
                    yi  = np.swapaxes(yi, 1, 0)
                    self.Xgrid = griddata( (gridlons.flatten(), gridlats.flatten()), xi.flatten(), (self.longitude, self.latitude) )
                    self.Ygrid = griddata( (gridlons.flatten(), gridlats.flatten()), yi.flatten(), (self.longitude, self.latitude) )
                    break
                except:
                    continue
        grid.close()
        if self.Xgrid == 0:
            self.Xgrid = np.nan
        if self.Ygrid == 0:
            self.Ygrid == np.nan

        if not all([np.isfinite(loc) for loc in [self.Xgrid, self.Ygrid]]):
            print('\tERROR: Unable to find sensible location for ({}E, {}N). Check if location is within grid domain! '.format(self.longitude, self.latitude))
            exit()

        return

    def createFile(self, cdl, outputfile):
        os.system('ncgen -b {} -o {}'.format(cdl, outputfile))
        return

    def dimensions(self, ds):
        xdim = 'x'
        ydim = 'y'
        for key in ds.dims.keys():
            if 'time' in key:
                timedim = key
            if 'eta' in key: #ROMS
                xdim = 'xi_rho'
                ydim = 'eta_rho'
            if 'lati' in key: # TP4_rean
                xdim = 'longitude'
                ydim = 'latitude'
        return timedim, xdim, ydim

    def replace_keywords(self, filein, fileout, worddict):
        '''
        replace keywords in a text file with given values from a dictionary

        filein: file name with keywords to be replaced
        fileout: file name for new file with replaced kewords
        worddict: dictionary of keywords and value

        johannesro@met.no
        '''
        lines = open(filein,'r').readlines()

        for keyword in sorted(worddict.keys(), key = len, reverse = True):
            newlines = []
            for line in lines:
                if keyword in line:
                    newlines.append(line.replace(keyword,worddict[keyword]))
                else:
                    newlines.append(line)
            lines = newlines

        out = open(fileout,'w')
        out.writelines(lines)
        out.close()

    def checkVars(self, out, frc, exclude = []):
        """
        Check if all required variables are on input forcing file.
        """
        has_vars = []
        for var in out.variables.keys():
            if len(out[var].shape) >= 3 :
                if not var in exclude: # keep these optional
                    has_vars.append(var in frc.variables.keys())

                    if not var in frc.variables.keys():
                        print('\tWARNING: {} not on file {}'.format(var, self.inputfile))
        if not all(has_vars):
            print('\tERROR: Required variables missing from input forcing file!!!')
            print('\tUnable to create forcing file.')
            exit()

    def extractForcing(self, ds, exclude = [], iniC = False):

        if iniC:
            ds.isel()
        out = xr.open_dataset(self.tmpfile, decode_times = False)
        timedimO, xdimO ,ydimO = self.dimensions(out)
        timedim, xdim ,ydim = self.dimensions(ds)
        if iniC:
            ds = ds.isel({timedim: 0})
        self.checkVars(out, ds, exclude = exclude  )

        # Fill variables:
        for var in out.variables.keys():
            if var in ds.variables.keys():
                if len(out[var].shape) >= 3:
                    tmp = ds[var].values
                    tmp = np.repeat(tmp, out[var].shape[-2]*out[var].shape[-1]).reshape(tmp.size, out[var].shape[-2], out[var].shape[-1])
                    out[var][:,:,:] = tmp

        out.to_netcdf(self.outputfile)
        timeunits = out[timedimO].units
        out.close()
        # expected_format = "seconds since 1970-01-01 00:00:00"
        # expected_format = "days since 1970-01-01 00:00:00"
        day_second = timeunits.split()[0]
        ref = np.datetime64(timeunits.split()[2])

        # Update time variable, xarray will not allow this:
        # also handle conversion of np.datetime64 to required format.
        with Dataset(self.outputfile, 'r+') as fid:
            var = fid.variables[timedimO]
            if 'day' in day_second:
                if ds[timedim].size > 1:
                    time_converted = np.array([ int(a -ref)/86400/10e8  for a in  ds[timedim].values[:] ])
                else:
                    time_converted = int(ds[timedim].values - ref)/86400/10e8
            elif 'second' in day_second:
                if ds[timedim].size > 1:
                    time_converted = np.array([ int(a -ref)/10e8  for a in  ds[timedim].values[:] ])
                else:
                    time_converted = int(ds[timedim].values - ref)/10e8

            var[:] =time_converted

class ForcingGenerator(Common):
    def __init__(self, raw_args = None, exclude = []):
        Common.__init__(self)

        '''
        ForcingGenerator can be run from another pyhton script with arguments parsed as a list [argname, argvalue, argname2, argvalue2]:
        ForcingGenerator(['--forcingType' , 'bulk', '--starttime', '2019-06-26', '--endtime', '2019-06-27', ...])
        '''
        self.tmpfile = None
        forcingType, self.inputfile, self.starttime, self.endtime, self.latitude, self.longitude, self.outputfile,  self.gridfile , self.Xgrid, self.Ygrid = self.parse_args(raw_args)

        print('')
        print('\tForcingGenerator')
        print('')
        if all([np.isfinite(self.longitude), np.isfinite(self.latitude)]):
            print('\tGenerating forcing for location ({}E,{}N) for the time span {} - {}.'.format(self.longitude, self.latitude, self.starttime, self.endtime))
        elif all([np.isfinite(self.Xgrid), np.isfinite(self.Ygrid)]):
            print('\tGenerating forcing for location ({},{}) for the time span {} - {}.'.format(self.Xgrid, self.Ygrid, self.starttime, self.endtime))


        print('\t{} Forcing will be retrieved from {}'.format(forcingType, self.inputfile))
        print('')

        if all([np.isfinite(self.longitude), np.isfinite(self.latitude), np.isnan(self.Xgrid), np.isnan(self.Ygrid)]):
            print('\tFinding index of ({}E,{}N)'.format(self.longitude, self.latitude))
            self.findIndex()
        else:
            print('\tRequire either lon/lat or xgrid/ygrid to be provided!')

        frc = xr.open_dataset(self.inputfile)
        frc, Ntimes = self.loadInput( frc)


        if 'bulk' in forcingType:
            self.__bulk(Ntimes)

        elif 'flux' in forcingType:
            self.__flux(Ntimes)

        self.extractForcing(frc, exclude = exclude)
        frc.close()
        os.system('rm {}'.format(self.tmpfile))
        
    def __bulk(self, Ntimes):
        '''
        generate bulk forcing!
        '''
        print('\tGenerate bulk forcing!')
        self.tmpfile = self.outputfile.format('bulk').replace('.nc', '_tmp.nc')
        self.outputfile = self.outputfile.format('bulk')
        self.replace_keywords('roms_bulkforce.cdl_KEYWORD', 'forcing.cdl', {'NTIMES': str(Ntimes)})
        self.createFile('forcing.cdl', self.tmpfile)
        return


    def __flux(self, Ntimes):
        '''
        generate flux forcing
        '''
        print('\tGenerate flux forcing!')
        self.tmpfile = self.outputfile.format('flux').replace('.nc', '_tmp.nc')
        self.outputfile = self.outputfile.format('flux')
        self.replace_keywords('roms_flux.cdl_KEYWORD', 'forcing.cdl', {'NTIMES': str(Ntimes)})
        self.createFile('forcing.cdl', self.tmpfile)
        return

class IniConditionGenerator(Common):
    def __init__(self, raw_args = None, exclude = []):
        Common.__init__(self)

        '''
        IniConditionGenerator can be run from another pyhton script with arguments parsed as a list [argname, argvalue, argname2, argvalue2]:
        IniConditionGenerator(['--forcingType' , 'ini', '--starttime', '2019-06-26', '--endtime', '2019-06-27', ...])
        '''
        self.tmpfile = None
        forcingType, self.inputfile, self.starttime, self.endtime, self.latitude, self.longitude, self.outputfile,  self.gridfile , self.Xgrid, self.Ygrid = self.parse_args(raw_args)

        print('')
        print('\tIniConditionGenerator')
        print('')
        if all([np.isfinite(self.longitude), np.isfinite(self.latitude)]):
            print('\tGenerating initial conditions for location ({}E,{}N) for first possible time in the time span {} - {}.'.format(self.longitude, self.latitude, self.starttime, self.endtime))
        elif all([np.isfinite(self.Xgrid), np.isfinite(self.Ygrid)]):
            print('\tGenerating initial conditions for location ({},{}) for first possible time in the time span {} - {}.'.format(self.Xgrid, self.Ygrid, self.starttime, self.endtime))


        print('\tInitial conditions will be retrieved from {}'.format(self.inputfile))
        print('')

        if all([np.isfinite(self.longitude), np.isfinite(self.latitude), np.isnan(self.Xgrid), np.isnan(self.Ygrid)]):
            print('\tFinding index of ({}E,{}N)'.format(self.longitude, self.latitude))
            self.findIndex()
        else:
            print('\tRequire either lon/lat or xgrid/ygrid to be provided!')

        ini = xr.open_dataset(self.inputfile)
        ini, Ntimes = self.loadInput( ini)

        if not 'ini' in forcingType:
            print('\tERROR!!! forcingType must be "ini". Given value is: {}'.format(forcingType))
            exit()
        self.__inicond(ini.s_rho.size)

        self.extractForcing(ini, exclude = exclude, iniC  = True)
        ini.close()
        os.system('rm {}'.format(self.tmpfile))

    def __inicond(self, Nlevels):
        '''
        generate initial conditions!
        '''
        print('\tGenerate initial conditions!')
        self.tmpfile = self.outputfile.format('ini').replace('.nc', '_tmp.nc')
        self.outputfile = self.outputfile.format('ini')
        self.replace_keywords('initial.cdl_KEYWORD', 'initial.cdl', {'NDEPTHS_RHO': str(int(Nlevels)), 'NDEPTHS_W':str(int(Nlevels+1))})
        self.createFile('initial.cdl', self.tmpfile)
        return
