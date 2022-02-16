import ini_and_frc

forcing_args = ['--latitude', '60',
                '--longitude', '7',
                '--starttime', '2017-02-05',
                '--endtime', '2017-05-01',
                '--forcingType', 'bulk',
                '--inputfile', '/lustre/storeB/project/fou/hi/cirfa/NorShelf_2017/Input/ocean_frc_2017.nc',
                '--outputfile', 'test_{}.nc'
                ]
ini_and_frc.ForcingGenerator(forcing_args, exclude = ['swrad', 'lwrad'])
ini_args = ['--latitude', '60',
                '--longitude', '7',
                '--starttime', '2017-01-07',
                '--endtime', '2017-01-10',
                '--forcingType', 'ini',
                '--inputfile', '/lustre/storeB/project/fou/hi/cirfa/NorShelf_2017/Input/ocean_clm_2017.nc',
                '--gridfile', '/lustre/storeB/project/fou/hi/norshelf/norshelf_etc/norshelf_2.4_vert_grd.nc',
                '--outputfile', 'test_{}.nc'
                ]
ini_and_frc.IniConditionGenerator(ini_args, exclude = ['rho', 'Hsbl', 'Hbbl', 'AKv', 'AKt', 'AKs'])
ini_args = ['--Xgrid', '419',
                '--Ygrid', '303',
                '--starttime', '2017-01-07',
                '--endtime', '2017-01-10',
                '--forcingType', 'ini',
                '--inputfile', '/lustre/storeB/project/fou/hi/cirfa/NorShelf_2017/Input/ocean_clm_2017.nc',
                '--gridfile', '/lustre/storeB/project/fou/hi/norshelf/norshelf_etc/norshelf_2.4_vert_grd.nc',
                '--outputfile', 'test2_{}.nc'
                ]
ini_and_frc.IniConditionGenerator(ini_args, exclude = ['rho', 'Hsbl', 'Hbbl', 'AKv', 'AKt', 'AKs'])
