/*
** svn $Id: lmd_test.h 939 2019-01-28 07:02:47Z arango $
*******************************************************************************
** Copyright (c) 2002-2019 The ROMS/TOMS Group                               **
**   Licensed under a MIT/X style license                                    **
**   See License_ROMS.txt                                                    **
*******************************************************************************
**
** Options for K-Profile Parameterization Test.
**
** Application flag:   COLUMN
** Input script:       roms_column.in
*/

#define UV_COR
#define UV_QDRAG
#define DJ_GRADPS
#define SPLINES_VDIFF
#define SPLINES_VVISC
#define NONLIN_EOS
#define SALINITY
#define SOLVE3D



#define LMD_MIXING
#ifdef LMD_MIXING
# define LMD_RIMIX
# define LMD_CONVEC
# define LMD_DDMIX
# define LMD_SKPP
# define LMD_BKPP
# define LMD_NONLOCAL
# define RI_SPLINES
#endif

#define LONGWAVE_OUT

#define ANA_INITIAL
#define ANA_BSFLUX
#define ANA_BTFLUX
#define ANA_GRID
