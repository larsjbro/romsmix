import re
import os
import glob
import csv
import pandas as pd
import subprocess
from datetime import datetime

import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import patheffects   # <-- THIS is required

import gsw
import xarray as xr  #used in get_latlon()

from netCDF4 import Dataset
FORCING_FOLDER = os.path.dirname(__file__)
ROOT = os.path.dirname(FORCING_FOLDER)
RUN_FOLDER = os.path.join(ROOT, 'Run')
EXTERNAL_FOLDER = os.path.join(ROOT, 'External')
INCLUDE_FOLDER = os.path.join(ROOT, 'Include')
RESULT_FOLDER = os.path.join(ROOT, 'Results')

SEC_PER_DAY = 24.0 * 3600.0
NANOSEC_PER_SECOND = 1e9

PLOT_DEPTH_MAX = -60

def mixing_regime_summary(romsfile, tke_threshold=1e-6):
    """
    Compute mixing regime summary for one ROMS experiment.

    Returns a dict with:
        mixed_layer_depth
        convective_fraction
        shear_fraction
        stable_fraction
    """

    f = Dataset(romsfile, 'r')

    # Extract vertical coordinate at rho-points 
    z_rho = f.variables['z_rho'][:,:,7,6]   # (nt, nz)
    z = z_rho[0,:]                          # (nz,)

    z_w   = f.variables['z_w'][:,:,7,6]     # (nt, nz+1)
    z_w_1d = z_w[0,:]                       # 1D (nz+1)

    # Extract variables
    rho = f.variables['rho'][:,:,7,6]       # (nt, nz)
    u   = f.variables['u'][:,:,7,6]
    v   = f.variables['v'][:,:,7,6]
    tke = f.variables['tke'][:,:,7,6]

    nt, nz = rho.shape
   
    Ri = gradient_richardson_number(rho, u, v, z)

    # Initialize with NaN so "unclassified" stays NaN
    stability = np.full_like(Ri, np.nan, dtype=float)

    valid = ~np.isnan(Ri)

    # Stability classification
    # 0 = convective, 1 = shear-driven, 2 = stable
    # Shear-driven: 0 <= Ri < 0.25
    stability[(Ri >= 0) & (Ri < 0.25) & valid] = 1

    # Stable: Ri >= 0.25
    stability[(Ri >= 0.25) & valid] = 2

    # Convective: Ri < 0
    stability[(Ri < 0) & valid] = 0

    # Fractions 
    total_points = nt * nz
    n_valid = np.sum(valid)
    n_undefined = np.sum(~valid)
    convective_fraction = np.nansum(stability == 0) / n_valid
    shear_fraction      = np.nansum(stability == 1) / n_valid
    stable_fraction     = np.nansum(stability == 2) / n_valid
    # Undefined fraction (relative to total)
    undefined_fraction = n_undefined / (n_valid + n_undefined)

    # Mixed layer depth (from TKE) 
    # For each time, find deepest depth where TKE > threshold
    #mld_times = compute_mld_tke(f, i=7, j=6, tke_factor=tke_threshold, adaptive=False)
    mld_times = compute_mld_density(f, i=7, j=6, drho_crit=0.03)

    mixed_layer_depth = np.nanmean(mld_times)

    f.close()

    return {
        "mixed_layer_depth": mixed_layer_depth,
        "convective_fraction": convective_fraction,
        "shear_fraction": shear_fraction,
        "stable_fraction": stable_fraction,
        "undefined_fraction": undefined_fraction,
    }


def csv_to_latex_matrix(csv_file,
                        caption="3×3 Mixing Regime Matrix",
                        label="tab:mixing_matrix"):
    """
    Create a LaTeX 3×3 matrix matching the experiment grid.
    Rows: Wind speeds (5, 10, 15)
    Cols: Cloud fractions (0.0, 0.5, 1.0)
    """

    df = pd.read_csv(csv_file)

    winds  = [5, 10, 15]
    clouds = [0.0, 0.5, 1.0]

    # Build matrix content
    matrix = [["" for _ in range(3)] for _ in range(3)]

    for i, wind in enumerate(winds):
        for j, cloud in enumerate(clouds):
            row = df[(df["wind_speed"] == wind) &
                     (df["cloud_fraction"] == cloud)]

            if len(row) == 1:
                mld    = f"{row['mixed_layer_depth_m'].values[0]:.1f}"
                conv   = f"{100*row['convective_fraction'].values[0]:.1f}"
                shear  = f"{100*row['shear_fraction'].values[0]:.1f}"
                stable = f"{100*row['stable_fraction'].values[0]:.1f}"

                matrix[i][j] = (
                    f"MLD={mld} m\\\\"
                    f"Conv={conv}\\\\"
                    f"Shear={shear}\\\\"
                    f"Stable={stable}"
                )
            else:
                matrix[i][j] = "N/A"

    # Build LaTeX table
    latex = "\\begin{table}[h!]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{c|ccc}\n"
    latex += "\\toprule\n"
    latex += "Wind [m/s] & Cloud 0.0 & Cloud 0.5 & Cloud 1.0 \\\\\n"
    latex += "\\midrule\n"

    for i, wind in enumerate(winds):
        latex += f"{wind} & {matrix[i][0]} & {matrix[i][1]} & {matrix[i][2]} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    # Save as .tex
    tex_file = os.path.splitext(csv_file)[0] + "_matrix.tex"
    with open(tex_file, "w") as f:
        f.write(latex)

    print(f"LaTeX matrix saved to {tex_file}")
    return latex


def csv_to_latex_table(csv_file,
                       caption="Mixing Regime Summary",
                       label="tab:mixing_summary"):
    """
    Convert mixing_summary CSV into a LaTeX table with:
      - Units in header
      - Escaped percent signs in header
      - Percent values without % sign
      - Cloud formatted with .1f
      - Row grouping by wind speed using \\multirow
      - Auto-save as .tex file
    """

    df = pd.read_csv(csv_file)

    # Format values
    df["Cloud_fmt"] = df["cloud_fraction"].map(lambda x: f"{x:.1f}")
    df["MLD"]       = df["mixed_layer_depth_m"].map(lambda x: f"{x:.1f}")
    df["Conv"]      = df["convective_fraction"].map(lambda x: f"{100*x:.1f}")
    df["Shear"]     = df["shear_fraction"].map(lambda x: f"{100*x:.1f}")
    df["Stable"]    = df["stable_fraction"].map(lambda x: f"{100*x:.1f}")

    winds = sorted(df["wind_speed"].unique())

    # Build LaTeX table manually to allow multirow
    latex = "\\begin{table}[h!]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{c|c|rrrr}\n"
    latex += "\\toprule\n"
    latex += "Wind [m/s] & Cloud & MLD [m] & Convective [\\%] & Shear [\\%] & Stable [\\%] \\\\\n"
    latex += "\\midrule\n"

    for wind in winds:
        df_w = df[df["wind_speed"] == wind]

        first = True
        for _, row in df_w.iterrows():
            cloud = row["Cloud_fmt"]
            mld   = row["MLD"]
            conv  = row["Conv"]
            shear = row["Shear"]
            stable= row["Stable"]

            if first:
                latex += (
                    f"\\multirow{{3}}{{*}}{{{wind}}} & "
                    f"{cloud} & {mld} & {conv} & {shear} & {stable} \\\\\n"
                )
                first = False
            else:
                latex += (
                    f" & {cloud} & {mld} & {conv} & {shear} & {stable} \\\\\n"
                )

        latex += "\\midrule\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    # Save to .tex file
    tex_file = os.path.splitext(csv_file)[0] + ".tex"
    with open(tex_file, "w") as f:
        f.write(latex)

    print(f"LaTeX table saved to {tex_file}")
    return latex


def create_3x3_mixing_summary_csv(csv_name="mixing_summary.csv", diurnal=True):
    """
    Creates a CSV summary for all 3x3 experiments.
    Rows: Wind Speeds (5, 10, 15)
    Cols: Cloud Values (0, 0.5, 1.0)
    Columns in CSV:
        wind_speed, cloud_fraction,
        mixed_layer_depth_m,
        convective_fraction,
        shear_fraction,
        stable_fraction
    """

    winds = [5, 10, 15]
    clouds = [0.0, 0.5, 1.0]

    # (cloud, wind) → experiment number
    diurnal_experiment_no = {
        ("0.0", 5): 30, ("0.0", 10): 28, ("0.0", 15): 32,
        ("0.5", 5): 46, ("0.5", 10): 50, ("0.5", 15): 57,
        ("1.0", 5): 49, ("1.0", 10): 53, ("1.0", 15): 54,
    }

    stable_experiment_no = {
        ("0.0", 5): 31, ("0.0", 10): 29, ("0.0", 15): 33,
        ("0.5", 5): 47, ("0.5", 10): 51, ("0.5", 15): 56,
        ("1.0", 5): 48, ("1.0", 10): 52, ("1.0", 15): 55,
    }

    experiment_no = diurnal_experiment_no if diurnal else stable_experiment_no

    num = experiment_no[("1.0", 15)]
    result_folder = os.path.join(RESULT_FOLDER, find_latest_experiment_folder(num))
    # Prepare CSV file
    outfile = csv_name.replace(".csv", "_diurnal.csv") if diurnal else csv_name.replace(".csv", "_stable.csv")
    outname = "mixing_heatmap_diurnal.png" if diurnal else "mixing_heatmap_stable.png"
    outfile = os.path.join(result_folder, outfile)
    outname = os.path.join(result_folder, outname)
    with open(outfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow([
            "wind_speed",
            "cloud_fraction",
            "mixed_layer_depth_m",
            "convective_fraction",
            "shear_fraction",
            "stable_fraction"
        ])

        # Loop through 3x3 grid
        for wind in winds:
            for cloud in clouds:

                cloud_str = f"{cloud:.1f}"
                num = experiment_no[(cloud_str, wind)]

                folder = os.path.join(RESULT_FOLDER, find_latest_experiment_folder(num))
                romsfile = os.path.join(folder, "roms_his.nc")

                if os.path.exists(romsfile):
                    summary = mixing_regime_summary(romsfile)

                    writer.writerow([
                        wind,
                        cloud,
                        summary["mixed_layer_depth"],
                        summary["convective_fraction"],
                        summary["shear_fraction"],
                        summary["stable_fraction"]
                    ])
                else:
                    writer.writerow([wind, cloud, "FILE_NOT_FOUND", "", "", ""])

    print(f"Mixing summary CSV saved to {outfile}")
    csv_to_latex_table(outfile)
    csv_to_latex_matrix(outfile)
    _create_3x3_mixing_heatmap(outfile, outname)


def _annotate_heatmap_cells(ax, data, fmt="{:.1f}"):
    """
    Annotate each cell of a heatmap with its numeric value.
    """
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                text = "–"
            else:
                text = fmt.format(val)

            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white",
                fontsize=11,
                fontweight="bold",
                path_effects=[
                    patheffects.Stroke(
                        linewidth=2, foreground='black'
                    ),
                    patheffects.Normal()
                ]
            )

def _create_3x3_mixing_heatmap(csv_file, outname="mixing_heatmap.png"):
    """
    Creates a 3×3 heatmap figure from the mixing_summary CSV.
    Rows: Wind Speeds (5, 10, 15)
    Cols: Cloud Values (0, 0.5, 1.0)
    """

    # Load CSV
    df = pd.read_csv(csv_file)

    winds = [15, 10, 5] #[5, 10, 15]
    clouds = [0.0, 0.5, 1.0]

    # Prepare matrices
    mixed_layer_depth = np.zeros((3, 3))
    convective_frac   = np.zeros((3, 3))
    shear_frac        = np.zeros((3, 3))
    stable_frac       = np.zeros((3, 3))

    # Fill matrices
    for i, wind in enumerate(winds):
        for j, cloud in enumerate(clouds):
            row = df[(df["wind_speed"] == wind) & (df["cloud_fraction"] == cloud)]
            if len(row) == 1:
                mixed_layer_depth[i, j] = row["mixed_layer_depth_m"].values[0]
                convective_frac[i, j]   = row["convective_fraction"].values[0]
                shear_frac[i, j]        = row["shear_fraction"].values[0]
                stable_frac[i, j]       = row["stable_fraction"].values[0]
            else:
                mixed_layer_depth[i, j] = np.nan
                convective_frac[i, j]   = np.nan
                shear_frac[i, j]        = np.nan
                stable_frac[i, j]       = np.nan

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Plot heatmaps
    def plot_heatmap(ax, data, title,  fmt="{:.1f}"):
        im = ax.imshow(data, cmap="viridis", origin="upper")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([f"{c:.1f}" for c in clouds])
        ax.set_yticklabels(winds)
        ax.set_xlabel("Cloud Fraction")
        ax.set_ylabel("Wind Speed [m/s]")
        ax.set_title(title, fontsize=14, fontweight="bold")

        _annotate_heatmap_cells(ax, data, fmt)
        fig.colorbar(im, ax=ax)

    plot_heatmap(axes[0,0], mixed_layer_depth, "Mixed Layer Depth [m]")
    plot_heatmap(axes[0,1], convective_frac * 100, "Convective Fraction [%]")
    plot_heatmap(axes[1,0], shear_frac * 100, "Shear Fraction [%]")
    plot_heatmap(axes[1,1], stable_frac * 100, "Stable Fraction [%]")

    # Save figure
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to {outname}")


def gradient_richardson_number(rho, u, v, z, g=9.81, rho0=1025):
    """
    Compute gradient Richardson number Ri_g from ROMS output.

    Parameters
    ----------
    rho : array (nt, nz)
        Density [kg/m^3]
    u, v : arrays (nt, nz)
        Horizontal velocities [m/s]
    z : array (nz)
        Vertical coordinate [m] (negative downward)
    g : float
        Gravity
    rho0 : float
        Reference density

    Returns
    -------
    Ri_g : array (nt, nz-1)
        Gradient Richardson number
    """

    # Vertical derivatives
    drho_dz = np.gradient(rho, z, axis=1)
    du_dz   = np.gradient(u,   z, axis=1)
    dv_dz   = np.gradient(v,   z, axis=1)

    # Buoyancy frequency
    N2 = -(g / rho0) * drho_dz  # Brunt Väisala

    # Vertical shear
    shear2 = du_dz**2 + dv_dz**2
    # Avoid division by zero
    shear2[shear2 == 0] = np.nan

    # Richardson number
    Ri_g = N2 / shear2

    return Ri_g


def now():
    """Returns current date time on isoformat: yyyy-mm-ddThhmmss"""
    return datetime.now().isoformat("T", "seconds").replace(':', '')


def extract_experiment_number(filename_string):
    """
    Extracts the digits following '_exp' in a ROMS simulation string.
    """
    # Search for '_exp' followed by one or more digits (\d+)
    match = re.search(r'_exp(\d+)', filename_string)
    
    if match:
        return int(match.group(1))
    
    return None


def print_forcing_info(filename, names=None):
    f = Dataset(filename, 'r')
    if names is None:
        names = ['swrad', 'shflux', 'swflux', 'sustr', 'svstr']
    for name in names:
        var = f.variables[name]
        avg = np.mean(var)
        vmin = np.min(var)
        vmax  = np.max(var)
        print(name, vmin, avg, vmax)
        print('')

def make_bulkforce_file(diurnal=True, sw_amplitude=800.0, cloud=0.0, u_wind=15.0):
    """Generate bulk forcing file"""
    # Generate empty forcing file and open for editing
    os.system('ncgen -b roms_bulkforce.cdl')
    f = Dataset(os.path.join(FORCING_FOLDER,'roms_bulkforce.nc'), 'a')

    # Make list of forcing variables to set (must match "roms_bulkfrc.cdl"!)
    frcvar = ['lwrad_down','cloud', 'Uwind', 'Vwind', 'Pair', 'Tair', 'Qair', 'rain', 'swrad']

    # Loop over variables and initialize to zero.
    #
    # NOTE! For more advanced input, edit yourself.
    #
    for varname in frcvar:
        print(varname)
        var = f.variables[varname]
        var[:] = 0.0

    # Setting lwrad_down to 400
    #f.variables['lwrad_down'][:,:,:] = 400.0

    # Setting swrad to constant value # See below for variable swrad
    #f.variables['swrad'][:,:,:] = 300.0    

    # Setting constant wind in x-direction
    f.variables['Uwind'][:,:,:] = u_wind

    # Setting cloud cover to 1
    f.variables['cloud'][:,:,:] = cloud

    # Setting constant humidity
    f.variables['Qair'][:,:,:] = 80.0

    # Setting constant temperature
    f.variables['Tair'][:,:,:] = 20.0 #10.0 old

    # Setting constant air pressure
    f.variables['Pair'][:,:,:] = 1020.0

    # Set time variable, the end time is the important one.
    #
    # NOTE! If time varying input, which requires more than two
    # time steps, remember to edit the size of "ocean_time" at
    # the start of "roms_frc.cdl".
    #
    ot = f.variables['ocean_time']
    # # ot.units == 'modified Julian day'
    num_time_steps = len(ot) #169
    #timevec = np.linspace(0.0,7*24*3600.0,num_time_steps) 
    timevec = np.linspace(0.0,7.0,num_time_steps)
    ot[:] = timevec

    # Setting shortwave diurnal cycle
    swrad = np.zeros_like(f.variables['swrad'][:])
    onedayfreq = 2*np.pi
    #sw_amplitude = -300.0  # Old
    
    # Antar at timevec er i DAGER. Faseforskyvning på 0.5 for å ha toppen ved middag (0.5 dager)
    phase_shift = 0.5
    
    sw_amplitude_clouds = (1.0 - 0.6 * cloud**3) *sw_amplitude
    print(f"sw_amplitude_clouds: {sw_amplitude_clouds}")
    if diurnal:
        for i in range(num_time_steps):
            swrad_value = sw_amplitude_clouds * np.cos(onedayfreq * (timevec[i] - phase_shift))
            # Setter negative verdier til null (ingen sol om natten)
            swrad_value = max(0.0, swrad_value) 

            swrad[i,:,:] = swrad_value * np.ones((14,12)) # 14x12 er antall celler i ditt grid
    else: #average diurnal
        swrad = np.maximum(sw_amplitude_clouds * np.cos(onedayfreq * (timevec - phase_shift)), 0).mean()
        print(f"Mean swrad: {swrad}")
        

    f.variables['swrad'][:,:,:] = swrad

    # Sync file to force write, then close.
    f.sync()
    f.close()
    # Rename bulk forcing file
    os.system('mv {0}/roms_bulkforce.nc {0}/roms_frc.nc'.format(FORCING_FOLDER))


def make_flux_force_file():
    """Generate flux forcing file"""
    # Generate empty forcing file and open for editing
    os.system('ncgen -b roms_flux.cdl')
    f = Dataset(os.path.join(FORCING_FOLDER, 'roms_flux.nc'), 'a')

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

    # Setting constant stress in x-direction  Typical range [0, 0.5]
    sustr = f.variables['sustr']
    #sustr[:] = 0.0
    #sustr[1] = 0.5
    sustr[:] = 0.1 

    # Setting constant cooling
    # sensible heat flux range [-100, 100]
    # Negative sign means flux coming out of the sea
    # Positive sign means flux into the sea
    shflux = f.variables['shflux']  
    #shflux[0:2] = 0.0
    #shflux[2:8] = 100.0
    #shflux[:] = -300.0
    shflux[:] = 100.0

    # Set time variable, the end time is the important one.
    #
    # NOTE! If time varying input, which requires more than two
    # time steps, remember to edit the size of "ocean_time" at
    # the start of "roms_frc.cdl".
    #
    ot = f.variables['ocean_time']
    # ot.units == 'seconds since 1970-01-01'
    #ot[:] = [0.0,7*24*3600.0] 
    num_time_steps = 169
    timevec = np.linspace(0.0, 7*24*3600.0, num_time_steps) 
    ot[:] = timevec

    # Setting shortwave diurnal cycle
    swrad = np.zeros_like(f.variables['swrad'][:])
    onedayfreq = 2*np.pi/(3600.*24)
    #sw_amplitude = -0.0  # Old
    sw_amplitude = 500.0  # new
    for i in range(num_time_steps):
        swrad[i,:,:] = sw_amplitude*np.cos(onedayfreq*timevec[i])*np.ones((14,12))
        
    swrad[np.where(swrad<0)] = 0.0


    f.variables['swrad'][:,:,:] = swrad

    # Sync file to force write, then close.
    f.sync()
    f.close()
    # Rename flux forcing file
    os.system('mv {0}/roms_flux.nc {0}/roms_frc.nc'.format(FORCING_FOLDER))


def run_roms(folder):
    """
    Parameters
    ----------
    folder: str
        Folder to store the results from the run
    
    Notes
    -----
    Run current compiled ROMS model using the flux forcing file 
    and saves the results to the given result folder.
    The build roms script is also copied to the given result folder.
    """
    # Copy flux forcing file to Run folder
    txt = subprocess.run(['cp', 
                          os.path.join(FORCING_FOLDER,'roms_frc.nc'),
                          os.path.join(RUN_FOLDER,'roms_frc.nc')], 
                         capture_output=True, text=True).stdout
    print("Finished generating forcing file: roms_frc.nc", txt)

    #Run romsS
    #os.system("../Run/romsS < ../External/roms_column.in > roms.log")
    os.system("{}/romsS < {}/roms_column.in > {}/roms.log".format(RUN_FOLDER, 
                                                                  EXTERNAL_FOLDER, 
                                                                  FORCING_FOLDER))
    print("Finished running romsS")
    move_results(folder)


def move_results(folder):
    print('Move input and results to ', folder)
    os.system('mkdir ' + folder)
    # log = subprocess.run(['mkdir', folder], capture_output=True, text=True).stdout
    print("Finished making folder:", folder)
    for fname in ['roms.log', 'roms_frc.nc', 'roms_his.nc', 'roms_rst.nc', ]:
        log = subprocess.run(['mv', 
                              os.path.join(FORCING_FOLDER, fname), 
                              os.path.join(folder, fname)], 
                              capture_output=True, text=True).stdout
        print('Moved ', fname)
    for fname in ['build_roms.sh']: 
        subprocess.run(['cp', 
                        os.path.join(ROOT, fname), 
                        os.path.join(folder, fname)], 
                        capture_output=True, text=True).stdout
        print('Copied ', fname)
    
    for subfolder in ['External','Include']:
        in_folder = os.path.join(ROOT, subfolder)
        out_folder = os.path.join(folder, subfolder)
        os.system('mkdir ' + out_folder)
        for fname in os.listdir(in_folder):
            subprocess.run(['cp', 
                        os.path.join(in_folder,fname), 
                        os.path.join(out_folder,fname)], 
                        capture_output=True, text=True).stdout
            print('Copied ', fname, ' to ', subfolder)
    

    #roms_log = subprocess.run(['../Run/romsS', '../External/roms_column.in'], capture_output=True, text=True).stdout
    #print("Finished running romsS:", roms_log)

def plot_density_difference_hovmuller(file1, file2, filename=None):
    """
    Plots the Hovmuller difference in density between two ROMS files.
    Calculates: delta_rho = rho_file1 - rho_file2
    """
    f1 = Dataset(file1, 'r')
    f2 = Dataset(file2, 'r')

    # 1. Get vertical coordinate and time from the first file
    # We assume the grids and time-steps are consistent between runs
    z_r = f1.variables['z_rho'][:,:,7,6]
    otime = f1.variables['ocean_time'][:] / (24 * 3600.0) # Convert to days
    
    # 2. Extract density from both files at the representative point (7, 6)
    rho1 = f1.variables['rho'][:,:,7,6]
    rho2 = f2.variables['rho'][:,:,7,6]
    
    # 3. Calculate the difference
    rho_diff = rho1 - rho2

    # 4. Prepare the grid for plotting
    nt, nz = np.shape(rho_diff)
    dt = np.array([otime,] * nz).transpose()

    # 5. Create the Plot
    plt.figure(figsize=(10, 6))
    
    # Use a diverging colormap (RdBu_r) so that 0 (no difference) is white
    # Red indicates file1 is denser; Blue indicates file1 is lighter
    limit = np.max(np.abs(rho_diff))
    #limit = 0.2
    levels = np.linspace(-limit, limit, 51)
    cf = plt.contourf(dt, z_r, rho_diff, levels=levels, cmap=cm.RdBu_r)

    # 6. Formatting
    plt.colorbar(cf, label=r'Density Difference $\Delta \rho$ [kg/m$^3$]')
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    plt.ylim([PLOT_DEPTH_MAX, 0])
    plt.grid(axis='x')
    #plt.title(f'Density Difference: {os.path.basename(file1)} - {os.path.basename(file2)}')
    
    if filename:
        plt.savefig(filename)
        plt.close()

    
    f1.close()
    f2.close()


def plot_Ri_hovmuller(romsfile, filename=None):
    # Open history file
    f = Dataset(romsfile, 'r')

    # Extract vertical coordinate at a single point
    z_rho = f.variables['z_rho'][:,:,7,6]

    # Extract density, u, v at same point
    rho = f.variables['rho'][:,:,7,6]
    u   = f.variables['u'][:,:,7,6]
    v   = f.variables['v'][:,:,7,6]

    # Extract time and convert to days
    otime = f.variables['ocean_time'][:]
    otime = otime/(24*3600.)
    ny = np.shape(rho)[1]
    dt = np.array([otime,]*ny).transpose()

    # Compute vertical gradients
    z = z_rho[0,:]
    Ri = gradient_richardson_number(rho, u, v, z, g=9.81, rho0=1025)
   
    # Open figure
    plt.figure()

    # Plot filled contours
    levels = np.linspace(-1, 1, 21)   # adjust as needed
    plt.contourf(dt, z_rho, Ri, levels=levels, cmap='RdBu_r')

    # Add info
    plt.colorbar(label='$R_i$')
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    plt.ylim([PLOT_DEPTH_MAX, 0])
    plt.grid(axis='x')

    # Save or show
    if filename:
        plt.savefig(filename)
        plt.close()

    # Close file
    f.close()


def plot_stability_hovmuller(romsfile, filename=None):
    # Open history file
    f = Dataset(romsfile, 'r')

    # Extract vertical coordinate at rho-points (nt, nz)
    z_rho = f.variables['z_rho'][:,:,7,6]
    z = z_rho[0,:]                 # 1D depth coordinate (nz,)

    # Extract density, u, v at same point (nt, nz)
    rho = f.variables['rho'][:,:,7,6]
    u   = f.variables['u'][:,:,7,6]
    v   = f.variables['v'][:,:,7,6]

    # Compute Richardson number using your function
    Ri = gradient_richardson_number(rho, u, v, z)

    # Extract time and convert to days
    otime = f.variables['ocean_time'][:]
    otime = otime/(24*3600.)
    nt, nz = rho.shape

    # Build time-depth mesh (same style as your TKE plot)
    dt = np.array([otime]*nz).transpose()   # (nt, nz)

    # --- Stability classification ---
    # 0 = convective, 1 = shear-driven, 2 = stable
    stability = np.zeros_like(Ri)
    stability[(Ri >= 0) & (Ri < 0.25)] = 1
    stability[Ri >= 0.25] = 2

    # Colormap for categories
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["red", "yellow", "blue"])
    labels = [r"Convective ($R_i<0$)", r"Shear ($0 \leq R_i<0.25$)", r"Stable ($R_i \geq 0.25$)"]

    # Open figure
    plt.figure(figsize=(10,6))

    # Plot categorical Hovmöller
    plt.pcolormesh(dt, z_rho, stability, cmap=cmap, shading='auto')

    # Colorbar with labels
    cbar = plt.colorbar(ticks=[0.33, 1, 1.66])
    cbar.ax.set_yticklabels(labels)

    # Add info
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    plt.ylim([PLOT_DEPTH_MAX, 0])
    plt.grid(axis='x')
    plt.title("Stability Classification (Richardson Number)")

    # Save or show
    if filename:
        plt.savefig(filename)
        plt.close()

    # Close file
    f.close()

def plot_tke_stability_hovmuller(romsfile, filename=None):
    # Open history file
    f = Dataset(romsfile, 'r')

    # --- Extract vertical coordinate at rho-points (nt, nz) ---
    z_rho = f.variables['z_rho'][:,:,7,6]
    z_w = f.variables['z_w'][:,:,7,6]
    z = z_rho[0,:]                 # 1D depth coordinate (nz,)

    # --- Extract variables at same point (nt, nz) ---
    rho = f.variables['rho'][:,:,7,6]
    u   = f.variables['u'][:,:,7,6]
    v   = f.variables['v'][:,:,7,6]
    tke = f.variables['tke'][:,:,7,6]


    Ri = gradient_richardson_number(rho, u, v, z)

    # --- Stability classification ---
    # 0 = convective, 1 = shear-driven, 2 = stable
    stability = np.zeros_like(Ri)
    stability[(Ri >= 0) & (Ri < 0.25)] = 1
    stability[Ri >= 0.25] = 2

    # --- Time axis ---
    otime = f.variables['ocean_time'][:]
    otime = otime/(24*3600.)        # convert to days
    nt, nz = rho.shape
    dt = np.array([otime]*nz).transpose()   # (nt, nz)
    dt_tke = np.array([otime]*(nz+1)).transpose() 
    # --- Open figure ---
    plt.figure(figsize=(10,6))

    # --- Background: TKE contourf (log-scale) ---
    levels_tke = list(range(-20, 1, 2))
   
    plt.contourf(dt_tke, z_w, np.log(tke), levels=levels_tke, cmap='Greys')

    # --- Overlay: Stability classification ---
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([
        (1.0, 0.0, 0.0, 0.4),   # red, convective, alpha=0.4
        (1.0, 1.0, 0.0, 0.4),   # yellow, shear, alpha=0.4
        (0.0, 0.0, 1.0, 0.4)    # blue, stable, alpha=0.4
    ])

    plt.pcolormesh(dt, z_rho, stability, cmap=cmap, shading='auto')

    # --- Colorbar for stability ---
    cbar = plt.colorbar(ticks=[0.33, 1, 1.66])
    cbar.ax.set_yticklabels([
        r"Convective ($R_i<0$)",
        r"Shear ($0≤R_i<0.25$)",
        r"Stable ($R_i≥0.25$)"
    ])

    # --- Labels and formatting ---
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    plt.ylim([PLOT_DEPTH_MAX, 0])
    plt.grid(axis='x')
    plt.title("TKE + Stability Overlay")

    # --- Save or show ---
    if filename:
        plt.savefig(filename)
        plt.close()

    f.close()


def plot_tke_hovmuller(romsfile,  filename=None):
    # Open history files and plot profiles
    f = Dataset(romsfile, 'r')

    # Get vertical coordinate
    z_w = f.variables['z_w'][:,:,7,6]

    # Get TKE
    tke = f.variables['tke'][:,:,7,6]

    # Get time
    otime = f.variables['ocean_time'][:]
    otime = otime/(24*3600.)
    ny = np.shape(tke)[1]
    dt = np.array([otime,]*ny).transpose()

    # Open figure
    plt.figure()

    # Plot filled contours
    levels = list(range(-20, 1, 2))
    plt.contourf(dt,z_w,np.log(tke),levels=levels)

    # Add info 
    plt.colorbar(label='log TKE [m^2/s^2]')
    #plt.colorbar(label='TKE difference [m^2/s^2]')
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    plt.ylim([PLOT_DEPTH_MAX, 0])
    plt.grid(axis='x')
    #plt.show()
    if filename:
        plt.savefig(filename)
        plt.close()
    # plt.show()

    # Close file
    f.close()

def plot_tke_difference_hovmuller(file1, file2, filename=None):
    """
    Plots the TKE Hovmuller difference 
    """
    f1 = Dataset(file1, 'r')
    f2 = Dataset(file2, 'r')

    # Get vertical coordinate
    z_w = f1.variables['z_w'][:,:,7,6]

    # Get TKE
    tke1 = f1.variables['tke'][:,:,7,6]
    tke2 = f2.variables['tke'][:,:,7,6]

    otime = f1.variables['ocean_time'][:]
    otime = otime/(24*3600.)
    ny = np.shape(tke1)[1]
    dt = np.array([otime,]*ny).transpose()
    
    # 4. Calculate the difference
    tke_diff = np.log(tke1) - np.log(tke2)

    # 6. Create the Plot
    plt.figure(figsize=(10, 6))
    
    # Diverging colormap: Red = Faster in File 1, Blue = Slower in File 1
    limit = np.max(np.abs(tke_diff))
    limit = 10
    levels = np.linspace(-limit, limit, 51)
    
    cf = plt.contourf(dt, z_w, tke_diff, levels=levels, cmap=cm.RdBu_r)

    # 7. Formatting
    plt.colorbar(cf, label=r'TKE Difference $\Delta log TKE$ [m^2/s^2]') 
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    plt.ylim([PLOT_DEPTH_MAX, 0])
    plt.grid(axis='x')
    
    if filename:
        plt.savefig(filename)
        plt.close()
    
    f1.close()
    f2.close()


def compute_mld_tke(f, i=7, j=6, tke_factor=1e-3, adaptive=True):
    """
    Compute TKE-based MLD using an adaptive threshold:
        threshold = tke_factor * max(TKE)
    Returns array of MLD depths (negative).
    """
    tke = f.variables["tke"][:, :, j, i]     # (nt, nz+1)
    z_w = f.variables["z_w"][:, :, j, i]     # (nt, nz+1)

    nt, nzp1 = tke.shape
    mld_tke = np.full(nt, np.nan)

    for t in range(nt):
        tke_t = tke[t, :]
        if adaptive:
            tke_max = np.nanmax(tke_t)
            threshold = tke_factor * tke_max
        else:
            threshold = tke_factor

        active = np.where(tke_t > threshold)[0]
        if len(active) > 0:
            mld_tke[t] = z_w[t, active[0]]   # deepest active turbulence, since z_w goes -200 → 0

    return mld_tke

def compute_mld_density(f, i=7, j=6, drho_crit=0.03):
    """
    Compute density-based MLD using the criterion:
        |rho(z) - rho(surface)| >= drho_crit

    Parameters
    ----------
    f : netCDF4.Dataset
        Open ROMS history/avg file.
    i, j : int
        Horizontal indices.
    drho_crit : float
        Density difference threshold (default 0.03 kg/m^3).

    Returns
    -------
    mld : array (nt,)
        Mixed layer depth (negative values).
    """

    rho = f.variables["rho"][:, :, j, i]     # (nt, nz)
    z_r = f.variables["z_rho"][:, :, j, i]   # (nt, nz)

    nt, nz = rho.shape
    mld = np.full(nt, np.nan)

    for t in range(nt):
        rho_sfc = rho[t, -1]                 # surface density (last index)
        diff = np.abs(rho[t, :] - rho_sfc)
        idx = np.where(diff >= drho_crit)[0]

        if len(idx) > 0:
            mld[t] = z_r[t, idx[-1]]         # first depth satisfying criterion

    return mld

def compute_entrainment_rate(mld, time_days):
    """
    Compute entrainment rate dMLD/dt in m/day.
    mld: array (nt,) of depths (negative)
    time_days: array (nt,) of time in days
    """
    # Use numpy gradient for smooth derivative
    entr = np.gradient(mld, time_days)
    return entr


def plot_density_hovmuller(romsfile, maxdensity, filename=None, MLD=False):
    """Open history files and plot Hovmuller density profiles,  optionally with MLD line."""
    f = Dataset(romsfile, 'r')

    # Get vertical coordinate
    z_r = f.variables['z_rho'][:,:,7,6]

    # Get density 
    rho = f.variables['rho'][:,:,7,6]

    # Get time
    otime = f.variables['ocean_time'][:]
    otime = otime/(24*3600.)
    nt, nz = rho.shape
    ny = np.shape(rho)[1]
    dt = np.array([otime,]*ny).transpose()

    # Check if we should limit range
    if maxdensity > 0.0:
        rho[rho>maxdensity] = np.nan
    
    # Compute MLD if requested
    ER = False
    if MLD:
        mld_rho = compute_mld_density(f, i=7, j=6, drho_crit=0.03)
        #mld_tke_a = compute_mld_tke(f, i=7, j=6, tke_factor=1e-3)
        mld_tke = compute_mld_tke(f, i=7, j=6, tke_factor=1e-6, adaptive=False)

    if MLD and ER:
        entr_rho = compute_entrainment_rate(mld_rho, otime)
        entr_tke = compute_entrainment_rate(mld_tke, otime)

        # Open figure
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10,10), sharex=True)

    
    print(np.shape(dt))
    print(np.shape(rho))
    print(np.shape(z_r))

    # Plot filled contours

    levels = np.linspace(25, 27, 51)
    im = ax1.contourf(dt, z_r, rho, levels=levels, cmap=cm.ocean_r)

    # Add info to ax1 (Hovmüller panel)
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Potential density anomaly [kg/m^3]')

    ax1.set_xlabel('Days')
    ax1.set_ylabel('Depth [m]')
    ax1.set_ylim([PLOT_DEPTH_MAX, 0])
    ax1.grid(axis='x')

    # plt.colorbar(label='Potential density anomaly [kg/m^3]')
    # plt.xlabel('Days')
    # plt.ylabel('Depth [m]')
    # plt.ylim([PLOT_DEPTH_MAX, 0])
    # plt.grid(axis='x')

    # --- Plot MLD line ---
    if MLD:
        ax1.plot(otime, mld_rho, 'r-', linewidth=2, label='Density MLD')
        ax1.plot(otime, mld_tke, 'b--', linewidth=2, label='TKE MLD')
        ax1.legend()
    if MLD and ER:
        ax2.plot(otime, entr_rho, 'r-', label='dMLD/dt (density)')
        ax2.plot(otime, entr_tke, 'b--', label='dMLD/dt (TKE)')
        ax2.axhline(0, color='k', linewidth=1)
        ax2.set_ylabel('Entrainment rate [m/day]')
        ax2.set_xlabel('Days')
        ax2.legend()

        # plt.plot(otime, mld_rho, 'r-', linewidth=2,
        #          label=r'MLD ($\Delta\rho = 0.03$ kg m$^{-3}$)')
        # plt.plot(otime, mld_tke, 'b--', linewidth=2,
        #          label=r'TKE MLD (threshold=1e-6)')
        # plt.plot(otime, mld_tke_a, 'g--', linewidth=2,
        #          label=r'TKE MLD (adaptive threshold)')
        #plt.legend()

    if filename:
        plt.savefig(filename)
        plt.close()
    # plt.show()

    # Close file
    f.close()

def plot_speed_difference_hovmuller(file1, file2, filename=None):
    """
    Plots the Hovmuller difference in horizontal speed between two ROMS files.
    Calculates: delta_speed = (sqrt(u1^2 + v1^2)) - (sqrt(u2^2 + v2^2))
    """
    f1 = Dataset(file1, 'r')
    f2 = Dataset(file2, 'r')

    # 1. Get vertical coordinate and time (assume consistency)
    # Using the same representative point (7, 6) as your density plots
    z_r = f1.variables['z_rho'][:, :, 7, 6]
    otime = f1.variables['ocean_time'][:] / (24 * 3600.0) # Days
    
    # 2. Extract Velocity components
    # Note: u and v are on staggered grids. For a point-comparison, 
    # we take the values at the same indices.
    u1 = f1.variables['u'][:, :, 7, 6]
    v1 = f1.variables['v'][:, :, 7, 6]
    u2 = f2.variables['u'][:, :, 7, 6]
    v2 = f2.variables['v'][:, :, 7, 6]
    
    # 3. Calculate Speed for both
    speed1 = np.sqrt(u1**2 + v1**2)
    speed2 = np.sqrt(u2**2 + v2**2)
    
    # 4. Calculate the difference
    speed_diff = speed1 - speed2

    # 5. Prepare grid for plotting
    nt, nz = np.shape(speed_diff)
    dt = np.array([otime,] * nz).transpose()

    # 6. Create the Plot
    plt.figure(figsize=(10, 6))
    
    # Diverging colormap: Red = Faster in File 1, Blue = Slower in File 1
    limit = np.max(np.abs(speed_diff))
    #limit = 0.03
    levels = np.linspace(-limit, limit, 51)
    
    cf = plt.contourf(dt, z_r, speed_diff, levels=levels, cmap=cm.RdBu_r)

    # 7. Formatting
    plt.colorbar(cf, label=r'Speed Difference $\Delta |U|$ [m/s]')
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    plt.ylim([PLOT_DEPTH_MAX, 0])
    plt.grid(axis='x')
    #plt.title(f'Speed Difference: {os.path.basename(file1)} - {os.path.basename(file2)}')
    
    if filename:
        plt.savefig(filename)
        plt.close()
    
    f1.close()
    f2.close()


def plot_speed_hovmuller(romsfile, filename=None):
    """Open history files and plot Hovmuller speed profiles"""
    # Open history files and plot profiles
    f = Dataset(romsfile, 'r')

    # Get vertical coordinate
    z_r = f.variables['z_rho'][:,:,7,6]

    # Get u,v
    u = f.variables['u'][:,:,7,6]  # N X M
    v = f.variables['v'][:,:,7,6]  # N X M 
    speed = np.sqrt(u**2 + v**2)
    # speed[speed>0.2] = 0.2  # LJB

    # Get time
    otime = f.variables['ocean_time'][:]
    otime = otime/(24*3600.)
    # dt = np.array([otime,]*42).transpose()  # LJB
    n = np.shape(speed)[1]
    dt = np.array([otime,]*n).transpose()  # LJB

    # Open figure
    plt.figure()

    # Plot filled contours
    #plt.contourf(dt,z_r,speed,levels=np.logspace(-6,0,25),norm=colors.LogNorm())
    # plt.contourf(dt,z_r,speed,25)  # LJB
    #levels = np.linspace(0, 0.4, 51)
    levels = 51
    plt.contourf(dt,z_r,speed,levels=levels)  # LJB

    # Add info 
    plt.colorbar(label='Speed [m/s]')
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    plt.ylim([PLOT_DEPTH_MAX, 0])
    plt.grid(axis='x')
    if filename:
        plt.savefig(filename)
        plt.close()
    # plt.show()

    # Close file
    f.close()


def plot_hodograph(romsfile, savefile=False, ext='.png'):
    # Check how many input files
    N = len(romsfile)

    # Plot hodograph
    plt.figure()

    for i in range(N):
        # Open history files and plot profiles
        f = Dataset(romsfile[i], 'r')

        # Get velocities at rho points
        u = f.variables['u_eastward'][:,-1,7,6]
        v = f.variables['v_northward'][:,-1,7,6]

        dt = 3600 # Output time step
        plt.plot([0],[0],'rd')
        
        if i==0:
            plt.plot(u, v, 'b-')
        else:
            plt.plot(u, v, 'g-')

        plt.axis('equal')
        if savefile:
            folder = os.path.dirname(romsfile[i])
            plt.savefig(os.path.join(folder, 'hodograph' +ext))
        # plt.show()

        # Close file
        f.close()


def get_latlon(i,j):
    

    # Open the ROMS grid file
    grid_ds = xr.open_dataset('roms_grd.nc')

    # Access the latitude and longitude arrays
    latitudes = grid_ds['lat_rho']
    longitudes = grid_ds['lon_rho']

    # Use the indices 7 and 6 to get the specific location
    # The dimensions are typically (eta_rho, xi_rho), corresponding to latitude and longitude
    lat_value = latitudes[i, j]
    lon_value = longitudes[i, j]

    print(f"Latitude: {lat_value.values}")
    print(f"Longitude: {lon_value.values}")


def get_density(file_path='roms_his.nc'):

    ds = xr.open_dataset(file_path)
    # Velg første tidspunkt og midtre lag for tetthet
    rho_data = ds['rho'].isel(ocean_time=0, s_rho=len(ds['s_rho']) // 2)


def calculate_roms_cell_volume(file_path: str):
    """
    Calculates the volume of each grid cell in a ROMS history file.

    The volume of a 3D ROMS grid cell is calculated by multiplying its
    horizontal area by its vertical thickness.

    Formula: Volume = (1 / pm) * (1 / pn) * Hz

    Where:
    - (1/pm) and (1/pn) are the grid spacing in the xi and eta directions.
    - pm and pn are the curvilinear coordinate metrics.
    - Hz is the vertical layer thickness at RHO-points.

    Args:
        file_path (str): The path to the ROMS history NetCDF file (e.g., 'roms_his.nc').
    """
    try:
        # Load the ROMS history file using xarray
        print(f"Loading dataset from: {file_path}")
        ds = xr.open_dataset(file_path)

        # Print a summary of the dataset to verify variables are present
        print("\nDataset loaded successfully. Available variables:")
        print(ds)

        # Check for the required variables. We now handle the case where Hz is missing.
        required_vars = ['pm', 'pn', 'z_w']
        for var in required_vars:
            if var not in ds.data_vars:
                raise KeyError(f"Required variable '{var}' not found in the dataset. "
                               "Please ensure it is included in your ROMS output.")

        # Extract the necessary variables. xarray automatically handles dimensions.
        pm = ds['pm']
        pn = ds['pn']
        if 'Hz' in ds:
            Hz = ds['Hz']
        else:
            z_w = ds['z_w']

            # If Hz is not in the file, calculate it from z_w.
            # z_w has N+1 vertical levels (from k=0 to N), while z_rho has N levels (k=1 to N).
            # The thickness of a rho layer is the difference between the z_w faces that bound it.
            print("\n'Hz' variable not found. Calculating vertical layer thickness (Hz) from 'z_w'...")
            # The slicing correctly aligns the dimensions for subtraction.
            Hz = z_w.diff(dim='s_w')
        
            # The new dimension 's_w' from the slicing now corresponds to 's_rho'.
            # We rename the dimension here to avoid the "cannot add coordinates" error.
            Hz = Hz.rename({'s_w': 's_rho'})

        # The core calculation: Volume = Horizontal Area * Vertical Thickness
        print("\nCalculating grid cell volume...")
        volume = (1 / pm) * (1 / pn) * Hz

        print("\nCalculation complete.")
        
        # Get the dimensions for robust indexing
        time_size = volume.sizes.get('ocean_time', 1)
        s_rho_size = volume.sizes.get('s_rho', 1)
        eta_rho_size = volume.sizes.get('eta_rho', 1)
        xi_rho_size = volume.sizes.get('xi_rho', 1)
        
        print(f"Shape of the resulting 'volume' array: ({time_size}, {s_rho_size}, {eta_rho_size}, {xi_rho_size})")

        # Let's inspect the data at a specific point for verification
        # Use robust indices to prevent out-of-bounds errors
        time_step = 0
        s_layer = s_rho_size // 2  # Middle layer
        eta_index = eta_rho_size // 2 # Middle of the eta dimension
        xi_index = xi_rho_size // 2  # Middle of the xi dimension

        print(f"\n--- Sample Data at Time Step {time_step}, Layer {s_layer}, Grid Cell ({xi_index}, {eta_index}) ---")
        
        # Select the sample data point
        sample_volume = volume.isel(ocean_time=time_step, s_rho=s_layer, eta_rho=eta_index, xi_rho=xi_index).values
        sample_pm = pm.isel(eta_rho=eta_index, xi_rho=xi_index).values
        sample_pn = pn.isel(eta_rho=eta_index, xi_rho=xi_index).values
        sample_Hz = Hz.isel(ocean_time=time_step, s_rho=s_layer, eta_rho=eta_index, xi_rho=xi_index).values

        print(f"pm: {sample_pm} (1/m)")
        print(f"pn: {sample_pn} (1/m)")
        print(f"Hz: {sample_Hz} (m)")
        print(f"Calculated volume: {sample_volume} (m^3)")

        # You can now save the volume to a new NetCDF file or plot it.
        # Example of saving the calculated volume to a new file:
        # volume.to_netcdf('roms_cell_volume.nc')

        # Example of a simple visualization for the first time step, mid-layer
        print("\nCreating a visualization of the cell volume for the first time step, mid-layer...")
        plt.figure(figsize=(10, 8))
        mid_layer_volume = volume.isel(ocean_time=time_step, s_rho=s_layer)
        mid_layer_volume.plot(x='xi_rho', y='eta_rho')
        plt.title(f'ROMS Grid Cell Volume at Layer {s_layer} (Time Step {time_step})')
        plt.xlabel('xi_rho')
        plt.ylabel('eta_rho')
        plt.show()

        print("\nScript finished successfully.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
    except KeyError as e:
        print(f"Error: A required variable is missing. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def robust_time_conversion(time_da: xr.DataArray):
    """
    Converts Xarray DataArray time to float (seconds and days) based on the 'units' attribute.
    
    Returns:
        tuple: (time_float_sec, time_days_da) 
               where time_float_sec is time in seconds (float)
               and time_days_da is a DataArray with time in days, used for coordinates.
    """
    

    time_values = time_da.values
    units = time_da.attrs.get('units', '').lower()
    
    # Case 1: Time is numeric and in DAYS (e.g., modified julian day)
    if 'day' in units and np.issubdtype(time_values.dtype, np.number):
        time_days_values = time_values.astype(float)
        time_float_sec = time_days_values * SEC_PER_DAY
        
    # Case 2: Time is numeric, assumed to be SECONDS
    elif np.issubdtype(time_values.dtype, np.number):
        time_float_sec = time_values.astype(float)
        time_days_values = time_float_sec / SEC_PER_DAY
        
    # Case 3: Time is datetime-like (e.g., datetime64[ns])
    else:
        # Convert nanoseconds to seconds
        time_float_sec = time_values.astype('int64').astype(float) / NANOSEC_PER_SECOND
        time_days_values = time_float_sec / SEC_PER_DAY

    # Create a DataArray with 'days' as coordinate for robust interpolation
    time_days_da = xr.DataArray(
        time_days_values,
        dims=time_da.dims,
        coords={time_da.dims[0]: time_days_values}
    )
    time_days_da.attrs['units'] = 'days'
    return time_float_sec, time_days_da


def plot_thermodynamic_fluxes(his_file: str, frc_file: str, filename: str=''):
    """
    Loads ROMS history and forcing files and plots the four main components 
    of the thermodynamic heat flux ($Q_H$) over time, averaged over the entire grid (W/m^2).
    
    The equation represented is:
    $Q_H = Q_{SW} + Q_{LW}^{\\text{net}} + Q_S + Q_L$
    
    The components plotted are:
    1. Shortwave (Q_SW)
    2. Net Longwave (Q_LW_net = Q_LW_down - Q_LW_out)
    3. Sensible Heat Flux (Q_S)
    4. Latent Heat Flux (Q_L)
    
    If Q_S and Q_L are not available separately, the Net Turbulent Flux (shflux) 
    is plotted as a single component instead.
    """
    try:
        print(f"Loading history file for SST: {his_file}")
        ds_his = xr.open_dataset(his_file)
        print(f"Loading forcing file for input data: {frc_file}")
        ds_frc = xr.open_dataset(frc_file)

        # --- TIME HANDLING ---
        _, his_time_days = robust_time_conversion(ds_his['ocean_time'])
        _, frc_time_days = robust_time_conversion(ds_frc['ocean_time'])
        
        # --- 1. Outgoing Longwave Radiation (Q_LW_out) Calculation ---
        epsilon = 0.97  # Emissivity
        sigma = 5.67e-8 # Stefan-Boltzmann constant (W m^-2 K^-4)

        # Retrieve Sea Surface Temperature (SST) in Kelvin
        sst_celsius = ds_his['temp'].isel(s_rho=-1, drop=True)
        sst_kelvin = sst_celsius + 273.15
        
        q_lw_out = epsilon * sigma * sst_kelvin**4
        q_lw_out = q_lw_out.assign_coords(ocean_time=his_time_days)
        
        # Interpolate Q_LW_out to forcing timesteps for calculation consistency
        q_lw_out_interp = q_lw_out.interp(
            ocean_time=frc_time_days.values, 
            method='linear', 
            kwargs={"fill_value": "extrapolate"}
        ).mean(dim=['eta_rho', 'xi_rho']) # Spatial average
        
        # --- 2. Main Components from Forcing File (Spatially Averaged) ---
        
        # Shortwave Radiation (Q_SW) - Positive = Heating
        q_sw = ds_frc['swrad'].mean(dim=['eta_rho', 'xi_rho'])
        q_sw = q_sw.assign_coords(ocean_time=frc_time_days)
        
        # Downward Longwave Radiation (Q_LW_down) - Positive = Heating
        q_lw_down = ds_frc['lwrad_down'].mean(dim=['eta_rho', 'xi_rho'])
        q_lw_down = q_lw_down.assign_coords(ocean_time=frc_time_days)
        
        # Net Longwave Flux (Q_LW_net) - Can be positive or negative
        q_lw_net = q_lw_down - q_lw_out_interp
        q_lw_net.name = 'Q_LW_net'

        # --- 3. Turbulent Flux Components (Q_S and Q_L) ---
        q_sensible = None
        q_latent = None
        q_turb_net = xr.zeros_like(q_sw) # Default Net Turbulent Flux
        
        turbulent_components_available = False
        
        if 'sensible' in ds_frc.variables and 'latent' in ds_frc.variables:
            # Separate Sensible and Latent Fluxes are available
            q_sensible = ds_frc['sensible'].mean(dim=['eta_rho', 'xi_rho'])
            q_sensible = q_sensible.assign_coords(ocean_time=frc_time_days)
            q_sensible.name = 'Q_S'
            
            q_latent = ds_frc['latent'].mean(dim=['eta_rho', 'xi_rho'])
            q_latent = q_latent.assign_coords(ocean_time=frc_time_days)
            q_latent.name = 'Q_L'
            
            q_turb_net = q_sensible + q_latent
            turbulent_components_available = True
            
        elif 'shflux' in ds_frc.variables:
            # Only Net Turbulent Flux (shflux) is available
            q_turb_net = ds_frc['shflux'].mean(dim=['eta_rho', 'xi_rho'])
            q_turb_net = q_turb_net.assign_coords(ocean_time=frc_time_days)
            q_turb_net.name = 'Q_S + Q_L' # Rename for plotting
            
        # --- 4. Net Total Heat Flux (Q_H) ---
        # Q_H = Q_SW + Q_LW_net + Q_Turb_net
        q_net_total = q_sw + q_lw_net + q_turb_net
        
        # --- Plotting ---
        
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot the four main components (W/m^2, grid average)
        q_sw.plot(ax=ax, label='$Q_{SW}$ (Shortwave)', color='red', linestyle='-')
        q_lw_net.plot(ax=ax, label='$Q_{LW}^{\\text{net}}$ (Net Longwave)', color='purple', linestyle='--')
        
        if turbulent_components_available:
            # Plot separate sensible and latent fluxes
            q_sensible.plot(ax=ax, label='$Q_S$ (Sensible)', color='orange', linestyle='-')
            q_latent.plot(ax=ax, label='$Q_L$ (Latent)', color='darkgreen', linestyle='-')
            
        elif 'shflux' in ds_frc.variables:
            # Plot net turbulent flux
            q_turb_net.plot(ax=ax, label='$Q_S + Q_L$ (Net Turbulent)', color='green', linestyle='-.')
        else:
            # If no turbulent fluxes are found, plot zero line for clarity (if not zero, the Q_net will be wrong)
            ax.plot(frc_time_days.values, q_turb_net.values, label='$Q_S + Q_L$ (Assumed Zero)', color='gray', linestyle=':')


        # Plot the total net flux
        q_net_total.plot(ax=ax, label='$Q_H$ (Net Total Heat Flux)', color='black', linewidth=3.0)

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
        
        title_eq = '$Q_H = Q_{SW} + Q_{LW}^{\\text{net}} + Q_S + Q_L$'
        ax.set_title(f'Thermodynamic Heat Flux Components (Grid Average)\nEquation: {title_eq}')
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Heat Flux (W/m$^2$)')
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            plt.close()
            print(f"Figure saved to {filename}")
        else:
            plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
    except KeyError as e:
        print(f"Error: A required variable is missing from a file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Debug: Error type was {type(e)}")


def calculate_model_heat_stats_2D_with_boundary(ds_his, rho0=1025.0, Cp=3985.0):
    """
    Calculates total heat content change and the rate of change from ROMS history data.

    Parameters
    ----------
    ds_his : xarray dataset
        ROMS history data
    rho0 : real scalar default 1025.0  
        Reference density of seawater (kg/m^3)
    Cp : real scalar, default 3985.0 
        Specific heat of seawater (J/(kg*K))

    Notes
    -----
    High accuracy since it uses Entire 2D Surface.
    """
    # Standardize time
    his_sec, his_days = robust_time_conversion(ds_his['ocean_time'])
    dt_his = np.mean(np.diff(his_sec))
    
    # 1. Calculate individual cell areas (dx * dy)
    # pm = 1/dx, pn = 1/dy -> area = 1/(pm*pn)
    area = 1.0 / (ds_his['pm'] * ds_his['pn']) 
    total_area = area.sum().item()
    
    # 2. Calculate layer thicknesses (Hz) and cell volumes
    z_w = ds_his['z_w']
    Hz = z_w.diff(dim='s_w').rename({'s_w': 's_rho'})
    volume = Hz * area # Results in (s_rho, eta_rho, xi_rho)
    
    # 3. Calculate Total Heat Content Change (Joules)
    temp = ds_his['temp']
    initial_temp = temp.isel(ocean_time=0, drop=True)
    temp_anomaly = temp - initial_temp

    volume_aligned = xr.DataArray(
        volume.values,
        coords=temp.coords,
        dims=temp.dims
    )
    
    # Total Heat = Sum(rho * Cp * delta_T * Volume)
    total_heat_joules = (rho0 * Cp * temp_anomaly * volume_aligned).sum(dim=['s_rho', 'eta_rho', 'xi_rho'])
    
    # 4. Rates and Normalization
    heat_content_per_m2 = total_heat_joules / total_area
    rate_of_change_watts = np.diff(total_heat_joules.values) / dt_his
    rate_of_change_per_m2 = rate_of_change_watts / total_area
    
    time_midpoints = (his_sec[:-1] + (dt_his / 2.0)) / SEC_PER_DAY #(24.0 * 3600.0) #86400.0
    rate_da = xr.DataArray(rate_of_change_per_m2, coords=[('ocean_time', time_midpoints)])
    
    return heat_content_per_m2.assign_coords(ocean_time=his_days), rate_da


def calculate_forcing_heat_stats_2D_with_boundary(ds_his):
    """
    Calculates integrated forcing flux and cumulative heat input from history data,
    including versions normalized per unit area (m^2).

    Notes
    -----
    Uses Entire 2D Surface, but gives somewhat lower values than calculate_forcing_heta_stats
    because of boundary effects. Use calculate_forcing_heat_stats_interior for better accuracy.
    """
    his_sec, his_days = robust_time_conversion(ds_his['ocean_time'])
    dt = np.mean(np.diff(his_sec))
    
    # 1. Grid Area
    area = 1.0 / (ds_his['pm'] * ds_his['pn'])
    total_area = area.sum(dim=['eta_rho', 'xi_rho']).item()
    
    # 2. Surface Flux (W/m^2)
    # shflux is the net surface heat flux (SW + LW + Latent + Sensible)
    net_flux_wm2 = ds_his['shflux'] 
    constant = 170 
    # 3. Total Integrated Forcing (Watts)
    # Multiply flux in every cell by that cell's area, then sum
    total_forcing_watts = (net_flux_wm2 * area ).sum(dim=['eta_rho', 'xi_rho'])
    
    # 4. Cumulative Energy Input (Joules)
    cumulative_heat_joules = (total_forcing_watts * dt).cumsum(dim='ocean_time')
    
    # 5. Normalization for comparison
    forcing_per_m2 = total_forcing_watts / total_area
    cumulative_per_m2 = cumulative_heat_joules / total_area
    
    return forcing_per_m2.assign_coords(ocean_time=his_days), \
           cumulative_per_m2.assign_coords(ocean_time=his_days)

def calculate_model_heat_stats_single_point(ds_his, rho0=1025.0, Cp=3985.0):
    """
    Calculates heat content change and rate of change per unit area (m^2).

    Parameters
    ----------
    ds_his : xarray dataset
        ROMS history data
    rho0 : real scalar default 1025.0  
        Reference density of seawater (kg/m^3)
    Cp : real scalar, default 3985.0 
        Specific heat of seawater (J/(kg*K))
    
    Notes
    -----
    Low accuracy since it uses a single point (6,7) and scales up.
    """
    temp = ds_his['temp']
    pm = ds_his['pm']
    pn = ds_his['pn']
    z_w = ds_his['z_w']
    
    # 1. Time handling
    his_sec, his_days = robust_time_conversion(ds_his['ocean_time'])
    dt_his = np.mean(np.diff(his_sec))
    
    # 1. Grid Area
    area0 = 1.0 / (ds_his['pm'] * ds_his['pn'])
    total_area0 = area0.sum(dim=['eta_rho', 'xi_rho']).item()
    
    # 2. Calculate area and total area
    # pm and pn are 1/dx and 1/dy
    try:
        cell_area = 1.0 / (pm.isel(eta_rho=6, xi_rho=7).item() * pn.isel(eta_rho=6, xi_rho=7).item())
        total_area = cell_area * (len(ds_his.eta_rho) * len(ds_his.xi_rho))
        # Area of a single cell
        #cell_area = 1.0 / (pm * pn)
        # Total horizontal area (Area * number of cells in eta and xi directions)
        #total_area = cell_area.sum(dim=['eta_rho', 'xi_rho'])
    except Exception:
        print("Advarsel: Grid-variablene pm/pn har uventede dimensjoner. Bruker standard 1x1 areal.")
        cell_area = 1.0
        total_area = 1.0 * (len(ds_his.eta_rho) * len(ds_his.xi_rho))
        
    # 3. Calculate Volume (Hz * cell_area)
    Hz = z_w.diff(dim='s_w').rename({'s_w': 's_rho'})
    volume_per_layer_aligned = xr.DataArray(
        Hz.values * cell_area,
        coords=temp.coords,
        dims=temp.dims
    )

    # 4. Calculate Total Heat Content (J) and normalize by area
    initial_temp = temp.isel(ocean_time=0, drop=True) 
    temp_anomaly = temp - initial_temp
    heat_content_per_cell = rho0 * Cp * temp_anomaly * volume_per_layer_aligned
    
    # Integrated total (J)
    total_heat_content = heat_content_per_cell.sum(dim=['eta_rho', 'xi_rho', 's_rho'])
    # Normalize to J/m^2
    heat_content_per_m2 = total_heat_content / total_area
    heat_content_per_m2 = heat_content_per_m2.assign_coords(ocean_time=his_days)

    # 5. Calculate Rate of Change (W) and normalize by area
    # (diff(J) / dt) results in Watts
    rate_of_change_total = np.diff(total_heat_content.values) / dt_his
    # Normalize to W/m^2
    rate_of_change_per_m2 = rate_of_change_total / total_area
    
    time_midpoints_days = (his_sec[:-1] + (dt_his / 2.0)) / SEC_PER_DAY  #(24.0 * 3600.0)
    rate_of_change_da = xr.DataArray(rate_of_change_per_m2, coords=[('ocean_time', time_midpoints_days)])

    return heat_content_per_m2, rate_of_change_da


def calculate_forcing_heat_stats_single_point(ds_his):
    """
    Calculates integrated forcing flux and cumulative heat input from history data,
    including versions normalized per unit area (m^2).

    Notes
    -----
    Low accurracy because a single point (6,7) is scaled up
    """
    
    his_sec, his_days = robust_time_conversion(ds_his['ocean_time'])
    pm = ds_his['pm']
    pn = ds_his['pn']

    cell_area = 1.0 / (pm.isel(eta_rho=6, xi_rho=7).item() * pn.isel(eta_rho=6, xi_rho=7).item())
    # 1. Coordinate and dimension handling
    num_eta = len(ds_his.eta_rho)
    num_xi = len(ds_his.xi_rho)
    total_area = cell_area * (num_eta * num_xi)
    
    # 2. Extract components (using the same representative cell as your original code)
    # Determine Net Heat Flux
    # Note: 'shflux' in ROMS history files often represents the net surface heat flux
    net_flux = ds_his['shflux'].isel(eta_rho=6, xi_rho=7).assign_coords(ocean_time=his_days)  # W/m^2
    
    # 4. Calculate Time Step
    dt = np.mean(np.diff(his_sec)) if len(his_sec) > 1 else (his_sec[1] - his_sec[0])
    
    # 5. Integrated Flux (Total and Per Area)
        
    # Total integrated flux (Watts)
    forcing_flux_sum = (net_flux * total_area).fillna(0.0)
    forcing_flux_sum = forcing_flux_sum.assign_coords(ocean_time=his_days)
    
    # Integrated flux per area (W/m^2)
    # This represents the average energy density entering the ocean surface
    forcing_flux_per_area = forcing_flux_sum / total_area

    # 6. Cumulative Heat Input (Total and Per Area)
    # Total cumulative input (Joules)
    cumulative_forcing_input = (forcing_flux_sum * dt).cumsum(dim='ocean_time')
    
    # Cumulative input per area (J/m^2)
    # This is the total energy accumulated per square meter over time
    cumulative_input_per_area = cumulative_forcing_input / total_area

    #return forcing_flux_sum, cumulative_forcing_input, 
    return forcing_flux_per_area, cumulative_input_per_area

def calculate_forcing_heat_stats_2D_interior(ds_his):
    """
    Calculates integrated forcing flux and cumulative heat input 
    integrating over the INTERIOR cells only (skipping the outer boundary).
    """
    his_sec, his_days = robust_time_conversion(ds_his['ocean_time'])
    dt = np.mean(np.diff(his_sec))
    
    # 1. Grid Area - Slice to [interior_eta, interior_xi]
    # Skipping the first and last index for both eta and xi
    area_full = 1.0 / (ds_his['pm'] * ds_his['pn'])
    area_interior = area_full.isel(eta_rho=slice(1, -1), xi_rho=slice(1, -1))
    
    # Calculate the total area of ONLY the interior cells
    total_area_interior = area_interior.sum(dim=['eta_rho', 'xi_rho']).item()
    
    # 2. Surface Flux (W/m^2) - Slice to interior
    # shflux dimensions are typically (ocean_time, eta_rho, xi_rho)
    net_flux_wm2_interior = ds_his['shflux'].isel(eta_rho=slice(1, -1), xi_rho=slice(1, -1))
    
    # 3. Total Integrated Forcing (Watts)
    # Integration only over interior pixels
    total_forcing_watts = (net_flux_wm2_interior * area_interior).sum(dim=['eta_rho', 'xi_rho'])
    
    # 4. Cumulative Energy Input (Joules)
    cumulative_heat_joules = (total_forcing_watts * dt).cumsum(dim='ocean_time')
    
    # 5. Normalization (Per Interior m^2)
    # We divide by the interior area so the units remain W/m^2 and J/m^2
    forcing_per_m2 = total_forcing_watts / total_area_interior
    cumulative_per_m2 = cumulative_heat_joules / total_area_interior
    
    return forcing_per_m2.assign_coords(ocean_time=his_days), \
           cumulative_per_m2.assign_coords(ocean_time=his_days)

def verify_heat_content(his_file: str, filename: str=''):
    """
    Calculates the total heat content in a ROMS simulation and compares it
    with the cumulative heat flux input from a forcing file.
    """
    try:
        ds_his = xr.open_dataset(his_file)

        # 1. Calculate Model Side
        model_heat_change, model_heat_rate = calculate_model_heat_stats_2D_with_boundary(ds_his)
        
        # 2. Calculate Forcing Side
        forcing_flux, cum_forcing = calculate_forcing_heat_stats_2D_interior(ds_his)

        # 3. Plotting
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))

        model_heat_change.plot(ax=axes[0], label='Model Heat Change', marker='o')
        cum_forcing.plot(ax=axes[0], label='Cumulative Forcing Input', marker='x', linestyle='--')
        axes[0].set_title('Comparison: Total Heat Content vs. Cumulative Flux')
        axes[0].legend()
        axes[0].grid(True)

        model_heat_rate.plot(ax=axes[1], label='Model Heat Rate (W/m^2)', marker='o')
        forcing_flux.plot(ax=axes[1], label='Total Forcing Flux (W/m^2)', marker='x', linestyle='--')
        axes[1].set_title('Comparison: Heat Change Rate')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            plt.close()
        # plt.show()
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
    except KeyError as e:
        print(f"Error: A required variable is missing from a file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Debug: Error type was {type(e)}")


def run_ncdiff(folder1, folder2):
    """
    Used ncdiff to calculate the difference between two netcdf-files and stores the result in folder2.
    """
    exp_num1 = extract_experiment_number(folder1)
    exp_num2 = extract_experiment_number(folder2)
    his_file1 = os.path.join(folder1, 'roms_his.nc')
    his_file2 = os.path.join(folder2, 'roms_his.nc')
    his_fileout = os.path.join(folder2, f'roms_his_diff{exp_num1}-{exp_num2}.nc')
    txt = subprocess.run(['ncdiff', his_file1, his_file2, his_fileout], 
                         capture_output=True, text=True).stdout
    print(f"Finished generating ncdiff file {his_fileout}: ", txt)


def compare_heat_content(folders: list[str], filename: str=''):
    """
    Calculates the total heat content in a ROMS simulation and compares it
    with the cumulative heat flux input from a forcing file.
    """
  
    try:
        # 3. Plotting
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
       
        for folder in folders:
            exp_num = extract_experiment_number(folder)

            his_file = os.path.join(folder, 'roms_his.nc')
            ds_his = xr.open_dataset(his_file)
           

            # 1. Calculate Model Side
            model_heat_change, model_heat_rate = calculate_model_heat_stats_2D_with_boundary(ds_his)
            
            # 2. Calculate Forcing Side
            forcing_flux, cum_forcing = calculate_forcing_heat_stats_2D_interior(ds_his)
     
            model_heat_change.plot(ax=axes[0], label=f'Model Heat Change exp.nr={exp_num}', marker='o')
            cum_forcing.plot(ax=axes[0], label=f'Cumulative Forcing Input exp.nr={exp_num}', marker='x', linestyle='--')
            model_heat_rate.plot(ax=axes[1], label=f'Model Heat Rate (W) exp.nr={exp_num}', marker='o')
            forcing_flux.plot(ax=axes[1], label=f'Total Forcing Flux (W) exp.nr={exp_num}', marker='x', linestyle='--')

        axes[0].set_title('Comparison: Total Heat Content vs. Cumulative Flux')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].set_title('Comparison: Heat Change Rate')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            plt.close()
        # plt.show()
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
    except KeyError as e:
        print(f"Error: A required variable is missing from a file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Debug: Error type was {type(e)}")

    
def plot_heat_flux_components(his_file: str, filename=None):
    """
    Plots individual heat flux components and their sum.
    Radiative: From ds_his
    Turbulent (Latent/Sensible): From ds_his
    """
    ds_his = xr.open_dataset(his_file)

    _, his_days = robust_time_conversion(ds_his['ocean_time'])
  
    # 1. Radiative Fluxes (Forcing)
    sw = ds_his['swrad'].isel(eta_rho=6, xi_rho=7).assign_coords(ocean_time=his_days)
    
    # Net Longwave calculation
    lw_net = ds_his['lwrad'].isel(eta_rho=6, xi_rho=7).assign_coords(ocean_time=his_days)
    
    # 2. Turbulent Fluxes (History - using ROMS standard variable names)
    # latent heat flux is often 'lhflux' and sensible is 'shflux' in history files
    latname = 'latent' #'lhflux'
    senname = "sensible" # 'shflux'
    latent = ds_his[latname].isel(eta_rho=6, xi_rho=7).assign_coords(ocean_time=his_days) if latname in ds_his else 0
    sensible = ds_his[senname].isel(eta_rho=6, xi_rho=7).assign_coords(ocean_time=his_days) if senname in ds_his else 0
    shflux = ds_his['shflux'].isel(eta_rho=6, xi_rho=7).assign_coords(ocean_time=his_days)

    # Calculate Net
    net_flux = sw + lw_net + latent + sensible

    # 4. Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(his_days, sw, label='Shortwave (Into Water)', color='gold', alpha=0.8)
    plt.plot(his_days, lw_net, label='Net Longwave', color='red', alpha=0.7)
    plt.plot(his_days, latent, label='Latent (From History)', color='blue', alpha=0.6)
    plt.plot(his_days, sensible, label='Sensible (From History)', color='green', alpha=0.6)
    plt.plot(his_days, shflux, label='shflux', marker='.', color='magenta', alpha=0.6)
    
    # Bold Net Flux
    plt.plot(his_days, net_flux, color='black', linewidth=2.5, label='NET HEAT FLUX')
    
    # Shade regions for clarity
    plt.fill_between(his_days, net_flux, 0, where=(net_flux > 0), color='orange', alpha=0.2, label='Heating Ocean')
    plt.fill_between(his_days, net_flux, 0, where=(net_flux < 0), color='cyan', alpha=0.2, label='Cooling Ocean')

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Surface Heat Flux Components (Positive = Heating Ocean)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Flux (W/m²)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    # plt.show()

def plot_temp_difference_hovmuller(file1, file2, filename=None):
    """
    Plots the Hovmuller difference in temperature between two ROMS files.
    Calculates: delta_temp = temp_file1 - temp_file2
    """
    f1 = Dataset(file1, 'r')
    f2 = Dataset(file2, 'r')

    # 1. Get vertical coordinate and time (assuming consistency between runs)
    # Using the representative point (eta_rho=7, xi_rho=6)
    z_r = f1.variables['z_rho'][:, :, 7, 6]
    otime = f1.variables['ocean_time'][:] / (24.0 * 3600.0) # Convert to days
    
    # 2. Extract Temperature from both files
    temp1 = f1.variables['temp'][:, :, 7, 6]
    temp2 = f2.variables['temp'][:, :, 7, 6]
    
    # 3. Calculate the difference
    temp_diff = temp1 - temp2

    # 4. Prepare the grid for plotting
    nt, nz = np.shape(temp_diff)
    dt = np.array([otime,] * nz).transpose()

    # 5. Create the Plot
    plt.figure(figsize=(10, 6))
    
    # Use a diverging colormap (RdBu_r)
    # Red = File 1 is warmer; Blue = File 1 is colder
    limit = np.max(np.abs(temp_diff))
    # Avoid zero-division if there is no difference
    limit = max(limit, 0.01) 
    levels = np.linspace(-limit, limit, 51)
    
    cf = plt.contourf(dt, z_r, temp_diff, levels=levels, cmap=cm.RdBu_r)

    # 6. Formatting
    cbar = plt.colorbar(cf)
    cbar.set_label(r'Temperature Difference $\Delta T$ [$^\circ$C]')
    
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    plt.ylim([PLOT_DEPTH_MAX, 0])
    plt.grid(axis='x')
    plt.title(f'Temperature Difference: {os.path.basename(file1)} - {os.path.basename(file2)}')
    
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
    
    f1.close()
    f2.close()

def plot_cons_temp_difference_hovmuller(file1, file2, filename=None):
    """
    Plots the Hovmuller difference in Conservative Temperature (CT) between two ROMS files.
    Calculates: delta_CT = CT_file1 - CT_file2
    """
    f1 = Dataset(file1, 'r')
    f2 = Dataset(file2, 'r')

    # Representative grid indices
    eta, xi = 7, 6

    # 1. Get coordinates and grid info
    z_r = f1.variables['z_rho'][:, :, eta, xi]
    otime = f1.variables['ocean_time'][:] / (24.0 * 3600.0)
    lat = 60.0 #f1.variables['lat_rho'][eta, xi]
    lon = 0.0 #f1.variables['lon_rho'][eta, xi]
    
    # 2. Calculate Conservative Temp for File 1
    pt1 = f1.variables['temp'][:, :, eta, xi]  # Potential Temp
    sp1 = f1.variables['salt'][:, :, eta, xi]  # Practical Salinity
    p1 = gsw.p_from_z(z_r, lat)                # Sea pressure
    sa1 = gsw.SA_from_SP(sp1, p1, lon, lat)    # Absolute Salinity
    ct1 = gsw.CT_from_pt(sa1, pt1)             # Conservative Temp

    # 3. Calculate Conservative Temp for File 2
    pt2 = f2.variables['temp'][:, :, eta, xi]
    sp2 = f2.variables['salt'][:, :, eta, xi]
    p2 = gsw.p_from_z(z_r, lat)                # Using z_r from file 1 for alignment
    sa2 = gsw.SA_from_SP(sp2, p2, lon, lat)
    ct2 = gsw.CT_from_pt(sa2, pt2)
    
    # 4. Calculate the difference
    ct_diff = ct1 - ct2

    # 5. Prepare grid for plotting
    nt, nz = np.shape(ct_diff)
    dt = np.array([otime,] * nz).transpose()

    # 6. Create the Plot
    plt.figure(figsize=(10, 6))
    
    limit = np.max(np.abs(ct_diff))
    limit = max(limit, 0.01) # Avoid range errors if files are identical
    levels = np.linspace(-limit, limit, 51)
    
    cf = plt.contourf(dt, z_r, ct_diff, levels=levels, cmap=cm.RdBu_r)

    # 7. Formatting
    cbar = plt.colorbar(cf)
    cbar.set_label(r'Conservative Temp Difference $\Delta \Theta$ [$^\circ$C]')
    
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    plt.ylim([PLOT_DEPTH_MAX, 0])
    plt.grid(axis='x')
    #plt.title(f'Conservative Temp Diff: {os.path.basename(file1)} vs {os.path.basename(file2)}')
    
    if filename:
        plt.savefig(filename)
        plt.close()
    
    f1.close()
    f2.close()

def plot_conservative_temp(romsfile, filename=None):
    """Open history files and plot conservative temperature profiles"""
    f = Dataset(romsfile, 'r')

    # Get vertical coordinate
    z_rho = f.variables['z_rho'][:,:,7,6]

    # Get density 
    rho = f.variables['rho'][:,:,7,6]

    # Get salt
    salt = f.variables['salt'][:,:,7,6]

    # Get temp 
    temp = f.variables['temp'][:,:,7,6]

    # Nlevels = [0,1,2]
    # Ncolumns = [0,1,2]

    # z_rho = np.transpose(z_rho, (1,0,2,3)).reshape((Nlevels, Ncolumns))
    # salt = np.transpose(salt, (1,0,2,3)).reshape((Nlevels, Ncolumns))
    # temp = np.transpose(temp, (1,0,2,3)).reshape((Nlevels, Ncolumns))

    # # Remove columns on land and reverse layers so that surface comes first
    # not_nan_ind = np.where(~np.isnan(z_rho[0,:]))[0]
    # z_rho = z_rho[::-1, not_nan_ind]
    # salt = salt[::-1, not_nan_ind]
    # temp = temp[::-1, not_nan_ind]

    # Convert salinity and temperature from ROMS to TEOS-10 standards
    p = gsw.conversions.p_from_z(z_rho, lat=60.0)
    SA = gsw.conversions.SA_from_SP(salt, p, lon=0.0, lat=60.0)
    CT = gsw.conversions.CT_from_pt(SA, temp)


    # Get time
    otime = f.variables['ocean_time'][:]
    otime = otime/(24*3600.)
    ny = np.shape(rho)[1]
    dt = np.array([otime,]*ny).transpose()

    # Check if we should limit range
    # if maxdensity > 0.0:
    #     rho[rho>maxdensity] = np.nan

    # Open figure
    
    plt.figure()
    print(np.shape(dt))
    print(np.shape(CT))
    print(np.shape(z_rho))

    # Plot filled contours
    # plt.contourf(dt, z_rho, salt, 200, cmap=cm.ocean_r)
    # plt.contourf(dt, z_rho, temp, 200, cmap=cm.ocean_r)
    levels = np.linspace(10, 22, 51)
    plt.contourf(dt, z_rho, CT, levels=levels, cmap=cm.ocean_r)
    plt.colorbar(label=r'Conservative Temperature [$C^{\circ}$]')
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    plt.ylim([PLOT_DEPTH_MAX, 0])
    plt.grid(axis='x')
    if filename:
        plt.savefig(filename)
        plt.close()
    # plt.show()

    # Close file
    f.close()

def plot_absolute_salinity(romsfile, filename=None):
    """Open history files and plot absolute salinity profiles"""
    f = Dataset(romsfile, 'r')

    # Get vertical coordinate
    z_rho = f.variables['z_rho'][:,:,7,6]

    # Get density 
    rho = f.variables['rho'][:,:,7,6]

    # Get salt
    salt = f.variables['salt'][:,:,7,6]

    # Get temp 
    temp = f.variables['temp'][:,:,7,6]

    # Convert salinity and temperature from ROMS to TEOS-10 standards
    p = gsw.conversions.p_from_z(z_rho, lat=60.0)
    SA = gsw.conversions.SA_from_SP(salt, p, lon=0.0, lat=60.0)


    # Get time
    otime = f.variables['ocean_time'][:]
    otime = otime/(24*3600.)
    ny = np.shape(rho)[1]
    dt = np.array([otime,]*ny).transpose()

    # Check if we should limit range
    # if maxdensity > 0.0:
    #     rho[rho>maxdensity] = np.nan

    # Open figure
    
    plt.figure()
    print(np.shape(dt))
    print(np.shape(SA))
    print(np.shape(z_rho))

    # Plot filled contours
    # plt.contourf(dt, z_rho, salt, 200, cmap=cm.ocean_r)
    # plt.contourf(dt, z_rho, temp, 200, cmap=cm.ocean_r)
    levels = np.linspace(35.16, 35.21, 51)

    plt.contourf(dt, z_rho, SA, levels=levels, cmap=cm.ocean_r)
    plt.colorbar(label=r'Absolute Salinity [$g/kg$]')
    plt.xlabel('Days')
    plt.ylabel('Depth [m]')
    plt.ylim([PLOT_DEPTH_MAX, 0])
    plt.grid(axis='x')
    if filename:
        plt.savefig(filename)
        plt.close()
    # plt.show()

    # Close file
    f.close()



def main(exp_name, useflux=True, diurnal=True, sw_amplitude=800.0, cloud=0.0, u_wind=15.0):
    """ 
    - Makes flux forcing file or bulk forcing file
    - Run current compiled ROMS model using the flux forcing file 
      and saves the results to the RESULT_FOLDER.
      The build roms script is also copied to the result folder
    - Plot hovmuller density profile and save the figure on pdf format
    """
    if useflux:
        make_flux_force_file()
    else:
        make_bulkforce_file(diurnal, sw_amplitude=sw_amplitude, cloud=cloud, u_wind=u_wind)
    folder = os.path.join(RESULT_FOLDER, now() + exp_name)
    run_roms(folder)
    ext = '.png'
    romsfile = os.path.join(folder, 'roms_his.nc')
    #forcing_file = os.path.join(folder, 'roms_frc.nc')
    plot_density_hovmuller(romsfile,
                           maxdensity=0.0,
                           filename=os.path.join(folder, 'density_hovmuller' +ext)
                           )
    plot_speed_hovmuller(romsfile,
                         filename=os.path.join(folder, 'speed_hovmuller' +ext)
                         )

    plot_conservative_temp(romsfile, filename=os.path.join(folder, 'conservative_temp' +ext))
    plot_absolute_salinity(romsfile, filename=os.path.join(folder, 'absolute_salinity' +ext))

    plot_hodograph([romsfile], savefile=True)
    #if useflux:
    verify_heat_content(romsfile, filename=os.path.join(folder, 'verify_heat_content' + ext))
    plot_heat_flux_components(romsfile, filename=os.path.join(folder, 'heat_flux_components' +ext))
    
    plt.show()


def find_latest_run_folder(exp_name: str) -> str:
    """Finds the path to the most recently created run folder with a given experiment name."""
    list_of_folders = glob.glob(os.path.join(RESULT_FOLDER, f'*-*{exp_name}*'))
    if not list_of_folders:
        raise FileNotFoundError(f"Found no foldere with '{exp_name}' in {RESULT_FOLDER}")
    latest_folder = max(list_of_folders, key=os.path.getctime)
    base_folder = os.path.basename(latest_folder)
    return base_folder

def find_latest_experiment_folder(exp_number: int) -> str:
    """Finds the path to the most recently created run folder with a given experiment number."""
    exp_name = f"_exp{exp_number}_"
    return find_latest_run_folder(exp_name)


def plots():
    # for folder in ['2025-03-09T164757no_F_cooling_0_xstress_0p5']: 
    # folders = [find_latest_run_folder('_exp15')]
    # folders = os.listdir(RESULT_FOLDER)
    #experiments = (2, 8)
    #experiments = (1, 9)
    experiments = list(range(28, 34)) + list(range(46, 58))
    folders = [find_latest_experiment_folder(i) for i in experiments]
    for folder in folders:
        if True: #folder.startswith('2025-09-24'):# 
            print(folder)
            root = os.path.join(RESULT_FOLDER, folder)
            romsfile = os.path.join(root, 'roms_his.nc')
            ext = '.png'
            plot_Ri_hovmuller(romsfile, filename=os.path.join(root, 'Ri_hovmuller' + ext))
            plot_stability_hovmuller(romsfile, filename=os.path.join(root, 'stability_hovmuller' + ext))
            #plot_tke_stability_hovmuller(romsfile, filename=os.path.join(root, 'tke_stability_hovmuller' + ext))

            # plot_hodograph([romsfile], savefile=True)
            # verify_heat_content(romsfile, filename=os.path.join(root, 'verify_heat_content' + ext))
            # plot_conservative_temp(romsfile, filename=os.path.join(root, 'conservative_temp' +ext))
            
            #plot_density_hovmuller(romsfile, maxdensity=0.0, filename=os.path.join(root, 'density_hovmuller' + ext))
            plot_density_hovmuller(romsfile, maxdensity=0.0, filename=os.path.join(root, 'density_hovmuller_mld' + ext), MLD=True)
            # plot_speed_hovmuller(romsfile, filename=os.path.join(root, 'speed_hovmuller' +ext))
            # plot_absolute_salinity(romsfile, filename=os.path.join(root, 'absolute_salinity' +ext))
            # plot_tke_hovmuller(romsfile, filename=os.path.join(root, 'tke_hovmuller' +ext))


            #plot_heat_flux_components(romsfile, filename=os.path.join(folder, 'heat_flux_components' +ext))
            # plot_thermodynamic_fluxes(romsfile, forcing_file, filename=os.path.join(root, 'thermodynamic_fluxes' + ext))
            #plt.show()
            #calculate_roms_cell_volume(romsfile)


def plot_kaihc(ext='.png'):
    kaifolder = find_latest_run_folder('_kaihc')
    print(kaifolder)
    root = os.path.join(RESULT_FOLDER, kaifolder)
    for experiment in os.listdir(root):
        if experiment.startswith('U10_'):
            folder = os.path.join(root, experiment)
            romsfile = os.path.join(folder,  f'KHC-his.nc-{experiment}')
            forcing_file = os.path.join(folder, f'roms_bulkforce.nc-{experiment}')
            plot_density_hovmuller(romsfile,
                           maxdensity=0.0,
                           filename=os.path.join(folder, 'density_hovmuller' +ext)
                           )
            plot_speed_hovmuller(romsfile,
                         filename=os.path.join(folder, 'speed_hovmuller' +ext)
                         )

            plot_hodograph([romsfile], savefile=True)
            plot_conservative_temp(romsfile, filename=os.path.join(folder, 'conservative_temp' +ext))
            plot_absolute_salinity(romsfile, filename=os.path.join(folder, 'absolute_salinity' +ext))


def extra_plots():
    #folder=os.path.join(RESULT_FOLDER, '2026-01-21T081848_exp24_Ninfo_1_strat_F_swrad_300_bulk_Uwind_5_cloud_0_Qair_80_Tair_10_Pair_1020')
    folder = os.path.join(RESULT_FOLDER, find_latest_run_folder('_exp28'))
    history_file = os.path.join(folder, 'roms_his.nc')
    
    # folder=os.path.join(RESULT_FOLDER, '2025-12-18_kaihc','U10_5-cloud_0-swrad_300')

    # plotte kai resultat
    #history_file = os.path.join(folder, 'KHC-his.nc-U10_5-cloud_0-swrad_300')

    verify_heat_content(history_file, filename=os.path.join(folder, 'verify_heat_content.png'))
    plot_heat_flux_components(history_file, filename=os.path.join(folder, 'heat_flux_components.png'))


def compare_results():
    experiments2compare = [
        # compare diurnal vs mean
        # (30, 31), 
        # (28, 29), 
        # (32, 33), 

        # (39, 38), 
        # (36, 37), 
        # (35, 34), 

        # (40, 41), 
        # (43, 42), 
        # (44, 45), 

        # compare increasing wind
        # (30, 28, 32), 
        # (39, 36, 35),
        # (40, 43, 44),

        # compare cloud and wind
        #(30, 36, 44),  # diurnal
        #(31, 37, 45),  # mean
        #(39,40) # diurnal 0.5,1
        
        #(30,46,49) #diurnal 0, 0.5, 1 fixed swcloudamplitude u=5
        (31,47,48) #mean 0, 0.5, 1 fixed swcloudamplitude u=5


        ]
    for experiments in experiments2compare:
        txt = ''.join([f'_exp{i}' for i in experiments])
        folders = [os.path.join(RESULT_FOLDER, find_latest_run_folder(f'_exp{i}_')) for i in experiments]
        
        compare_heat_content(folders, filename=os.path.join(folders[-1], f'compare_heat_content{txt}.png'))

def compare_ncdiff():
    # døgnsyklys
    #        u5, u10, u15 
    # c0.0 : 30,  28,  32
    # c0.5 : 46,  50,  57
    # c1.0 : 49,  53,  54
    # # ikke døgnsyklus
    #        u5, u10, u15 
    # c0.0 : 31,  29,  33
    # c0.5 : 47,  51,  56
    # c1.0 : 48,  52,  55


    experiments2compare = [
        (46, 49), # cloud 0.5 vs 1.0, vind 5, døgnsyklus
        (46, 50), # cloud 0.5, vind 5 vs 10, døgnsyklus

        (47, 48), # cloud 0.5 vs 1.0, vind 5
        (47, 51), # cloud 0.5, vind 5 vs 10

        (30, 31), # døgnsyklus vs ikke, vind 5 m/s, cloud 0
        (28, 29), # døgnsyklus vs ikke, vind 10 m/s, cloud 0
        (32, 33), # døgnsyklus vs ikke, vind 15 m/s, cloud 0

        (46, 47), # døgnsyklus vs ikke, vind 5 m/s, cloud 0.5
        (50, 51), # døgnsyklus vs ikke, vind 10 m/s, cloud 0.5
        (57, 56), # døgnsyklus vs ikke, vind 15 m/s, cloud 0.5

        (49, 48), # døgnsyklus vs ikke, vind 5 m/s, cloud 1.0
        (53, 52), # døgnsyklus vs ikke, vind 10 m/s, cloud 1.0
        (54, 55), # døgnsyklus vs ikke, vind 15 m/s, cloud 1.0


        # compare increasing cloud and wind
        (30, 50),
        (50, 54),

        (31, 51),
        (51, 55),

        ]
    for experiments in experiments2compare:
        folders = [os.path.join(RESULT_FOLDER, find_latest_run_folder(f'_exp{i}')) for i in experiments]
        folder1, folder2 = folders
        #run_ncdiff(folder1, folder2)

        exp_num1, exp_num2 = experiments
        ext = '.png'
        difftxt = f"diff{exp_num1}-{exp_num2}"
        romsfile = os.path.join(folder2, f'roms_his_{difftxt}.nc')
        #forcing_file = os.path.join(folder, 'roms_frc.nc')
        romsfile1 = os.path.join(folder1, 'roms_his.nc')
        romsfile2 = os.path.join(folder2, 'roms_his.nc')
        plot_density_difference_hovmuller(romsfile1, romsfile2, filename=os.path.join(folder2, f'density_hovmuller_{difftxt}' +ext))
        plot_speed_difference_hovmuller(romsfile1, romsfile2, filename=os.path.join(folder2, f'speed_hovmuller_{difftxt}' +ext))
        plot_cons_temp_difference_hovmuller(romsfile1, romsfile2, filename=os.path.join(folder2, f'conservative_temp_{difftxt}' +ext))
        plot_tke_difference_hovmuller(romsfile1, romsfile2, filename=os.path.join(folder2, f'tke_hovmuller_{difftxt}' +ext))
        # plot_density_hovmuller(romsfile,
        #                     maxdensity=0.0,
        #                     filename=os.path.join(folder2, f'density_hovmuller_{difftxt}' +ext)
        #                     )
        # plot_speed_hovmuller(romsfile,
        #                     filename=os.path.join(folder2, f'speed_hovmuller_{difftxt}' +ext)
        #                     )

        # plot_conservative_temp(romsfile, filename=os.path.join(folder2, f'conservative_temp_{difftxt}' +ext))
    

def create_3x3_experiment_grid(plot_name="temp_hovmuller.png", diurnal=True):
    """
    Creates a 3x3 grid of PNG images.
    Rows: Wind Speeds (5, 10, 15)
    Cols: Cloud Values (0, 0.5, 1.0)
    """
    winds = [15, 10, 5]
    clouds = [0.0, 0.5, 1.0]
                     # (c, u)
    diurnal_experiment_no = {("0.0", 5): 30, ("0.0", 10): 28, ("0.0", 15): 32,
                            ("0.5", 5): 46, ("0.5", 10): 50, ("0.5", 15): 57,
                            ("1.0", 5): 49, ("1.0", 10): 53, ("1.0", 15): 54,
                     }
    stable_experiment_no = {("0.0", 5): 31, ("0.0", 10): 29, ("0.0", 15): 33,
                             ("0.5", 5): 47, ("0.5", 10): 51, ("0.5", 15): 56,
                             ("1.0", 5): 48, ("1.0", 10): 52, ("1.0", 15): 55,
                     }
    experiment_no = diurnal_experiment_no if diurnal else stable_experiment_no
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, wind in enumerate(winds):
        for j, cloud in enumerate(clouds):

            cloud_str = f"{cloud:.1f}"
            num = experiment_no[(cloud_str, wind)]
            folder = os.path.join(RESULT_FOLDER, find_latest_experiment_folder(num)) 

            ax = axes[i, j]
            
            img_path = os.path.join(folder, plot_name)
            
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.set_title(f"Wind: {wind} m/s | Cloud: {cloud}", fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f"Image not found:\nU={wind}, C={cloud}", ha='center')
            
            # Remove ticks for a cleaner table look
            ax.set_xticks([])
            ax.set_yticks([])

    # Add global Labels
    fig.suptitle(f"Experiment Matrix: {plot_name}", fontsize=20, y=0.95)
    num = experiment_no[("1.0", 15)]
    folder = os.path.join(RESULT_FOLDER, find_latest_experiment_folder(num))
    # Label the axes of the 3x3 grid
    for ax, wind in zip(axes[:,0], winds):
        ax.set_ylabel(f"Wind {wind} m/s", rotation=90, size='large', fontweight='bold')
    for ax, cloud in zip(axes[-1,:], clouds):
        ax.set_xlabel(f"Cloud {cloud}", size='large', fontweight='bold')
    if diurnal:
        outfile = plot_name.replace('.png', '_diurnal_matrix.png')
    else:
        outfile = plot_name.replace('.png', '_stable_matrix.png')
    filename = os.path.join(folder, outfile)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Matrix saved to {filename}")

def make_summary():
    """Make summary plots and mixing summary table for each 3 X 3 experiment
    
    The summary plots  and mixing summary table is stored in the 
    experiment-folder no 54 and 55.
    """
    for name in [
        'speed_hovmuller', 
        "stability_hovmuller",
        "tke_hovmuller",
        "density_hovmuller",
        "density_hovmuller_mld",
        "hodograph",
        "Ri_hovmuller",
        "conservative_temp",
        ]:
        create_3x3_experiment_grid(name + ".png", diurnal=True)
        create_3x3_experiment_grid(name + ".png", diurnal=False)
    
    create_3x3_mixing_summary_csv(csv_name="mixing_summary.csv", diurnal=True)
    create_3x3_mixing_summary_csv(csv_name="mixing_summary.csv", diurnal=False)

if __name__ == "__main__":
    # main(exp_name='_exp57_Ninfo_1_strat_F_swradmax_740_bulk_Uwind_15_cloud_0p5_Qair_80_Tair_20_Pair_1020',
    #      useflux=False, diurnal=True, sw_amplitude=800, cloud=0.5, u_wind=15.0)

    make_summary()

    #plots()
    #compare_ncdiff()

    # extra_plots()
    # compare_results()
    # plot_kaihc()

    # print_forcing_info(os.path.join(RESULT_FOLDER, 
    #                                '2025-03-30T164056_exp6_strat_no_F_cooling_m100_xstress_0p1',
    #                                'roms_frc.nc'))
