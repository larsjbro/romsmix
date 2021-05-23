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
#undef UV_ADV
#define UV_QDRAG
#undef NONLIN_EOS
#define SALINITY
#define SOLVE3D

#ifdef COLUMN_CONST_MIX
# define ANA_VMIX
#endif

#if defined(COLUMN_STRAT) || defined(COLUMN_NO_STRAT) 
# define ANA_INITIAL
#endif

#ifdef LMD_MIXING
# define SPLINES_VDIFF
# define SPLINES_VVISC
# define LMD_RIMIX
# define LMD_CONVEC
# define LMD_DDMIX
# define LMD_SKPP
# define LMD_NONLOCAL
# define RI_SPLINES
#endif

#ifdef GLS_MIXING
# define SPLINES_VDIFF
# define SPLINES_VVISC
# define CRAIG_BANNER
# define CANUTO_A           
#endif

#define LONGWAVE_OUT

#define ANA_BSFLUX
#define ANA_BTFLUX
#define ANA_GRID
