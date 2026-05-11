"""
PyLag Grid Resolution Testing Script

Tests how temporal grid resolution affects particle tracking accuracy by:
1. Generating synthetic K(z,t) data on a fine temporal grid
2. Interpolating that to a coarse temporal grid (simulating downsampled input data)
3. Running PyLag simulations driven by each dataset
4. Computing diagnostics and visualisations comparing the outputs

The spatial (depth) grid is held constant between runs. Only the time axis
density of the forcing data differs, reflecting a real-world scenario where
you might have high-frequency model output vs. sub-sampled archive data.
"""

import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from configparser import ConfigParser
import subprocess
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Try importing PyLag components
try:
    from pylag.processing.input import create_initial_positions_file_single_group
    PYLAG_AVAILABLE = True
except ImportError:
    PYLAG_AVAILABLE = False
    print("Warning: PyLag not available. Simulation steps will be skipped.")


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generate synthetic K(z, t) data on a fixed depth grid.

    The depth grid does NOT change between the coarse and fine datasets.
    Only the temporal sampling frequency differs.

    A smooth, physically plausible diffusivity field is produced:
      K(z, t) = base_profile(z)  *  temporal_modulation(t)  *  spatial_noise(z)

    where base_profile has a double-peak shape (high near surface and bed,
    low in the pycnocline) and temporal_modulation mimics a semi-diurnal tidal
    signal.
    """

    def __init__(self, z_min=0.0, z_max=50.0, t_duration_hours=24.0):
        self.z_min = z_min
        self.z_max = z_max
        self.t_duration = t_duration_hours * 3600.0  # seconds

    # ------------------------------------------------------------------
    def _smooth_noise(self, n_points, seed=42):
        """Return a smooth multiplicative noise array of length n_points."""
        np.random.seed(seed)
        n_knots = max(4, n_points // 10)
        knot_idx = np.linspace(0, n_points - 1, n_knots)
        knot_vals = np.random.uniform(0.6, 1.4, n_knots)
        interp = interp1d(knot_idx, knot_vals, kind='cubic',
                          bounds_error=False, fill_value='extrapolate')
        noise = interp(np.arange(n_points))
        return np.clip(noise, 0.3, 2.0)

    # ------------------------------------------------------------------
    def _base_profile(self, z_array):
        """
        Depth-dependent base diffusivity.

        High at surface and bed, suppressed in the middle (stratification).
        Units: m² s⁻¹
        """
        z_norm = (z_array - self.z_min) / (self.z_max - self.z_min)
        # sin² gives a mid-water-column minimum; offset gives a realistic floor
        return 1e-4 + 1e-2 * (1.0 - np.sin(np.pi * z_norm)) ** 2

    # ------------------------------------------------------------------
    def _temporal_modulation(self, t_array):
        """
        Semi-diurnal tidal modulation plus a weak diurnal signal.
        Returns a 1-D array, same length as t_array.
        """
        t_norm = t_array / self.t_duration
        return 1.0 + 0.5 * np.sin(4 * np.pi * t_norm) \
                   + 0.2 * np.sin(2 * np.pi * t_norm)

    # ------------------------------------------------------------------
    def _diffusivity_field(self, z_array, t_array):
        """
        Full K(z, t) array of shape (n_times, n_z).
        """
        base   = self._base_profile(z_array)            # (n_z,)
        noise  = self._smooth_noise(len(z_array))       # (n_z,)
        tmod   = self._temporal_modulation(t_array)     # (n_t,)

        # Outer product: each time slice = base * noise, scaled by tmod[i]
        K = np.outer(tmod, base * noise)                # (n_t, n_z)
        return np.clip(K, 1e-6, None)                   # enforce positivity

    # ------------------------------------------------------------------
    @staticmethod
    def _zi_from_z(z_array):
        """
        Derive the N+1 interface levels from N cell-centre levels.

        Interfaces are placed at midpoints between successive centres, with the
        bottom boundary extended half a cell below z[0] and the top boundary
        extended half a cell above z[-1], matching real GOTM convention
        (z = 0 at the sea bed).

        Parameters
        ----------
        z_array : 1-D array of cell-centre heights above sea bed (m), length N

        Returns
        -------
        zi_array : 1-D array of interface heights above sea bed (m), length N+1
        """
        midpoints = 0.5 * (z_array[:-1] + z_array[1:])
        zi_array  = np.concatenate([
            [z_array[0]  - 0.5 * (z_array[1]  - z_array[0])],
            midpoints,
            [z_array[-1] + 0.5 * (z_array[-1] - z_array[-2])],
        ])
        # Clamp so the bed interface is never below 0
        zi_array[0] = max(zi_array[0], 0.0)
        return zi_array

    # ------------------------------------------------------------------
    def create_dataset(self, z_array, dt_seconds,
                       start_time='2010-06-01 00:00:00'):
        """
        Build an xarray Dataset matching the GOTM NetCDF convention.

        Dimensions
        ----------
        z  : N cell-centre heights above sea bed — stored as a coordinate for
             reference; scalar tracers would live here in a full GOTM file.
        zi : N+1 interface heights above sea bed — the dimension on which
             turbulent fluxes such as nuh are defined, as in real GOTM output.

        Variables
        ---------
        nuh : eddy diffusivity on (time, zi)

        Parameters
        ----------
        z_array    : 1-D array of cell-centre heights above sea bed (m)
        dt_seconds : temporal sampling interval in seconds
        start_time : ISO string for the first time stamp
        """
        n_times  = int(self.t_duration / dt_seconds) + 1
        t_secs   = np.arange(n_times) * dt_seconds
        times    = [datetime.fromisoformat(start_time) + timedelta(seconds=float(s))
                    for s in t_secs]

        zi_array = self._zi_from_z(z_array)                    # (N+1,)
        K        = self._diffusivity_field(zi_array, t_secs)   # (n_t, N+1)

        z4  = xr.DataArray(
            z_array[:, None, None], 
            dims=("z", "lat", "lon")
        ).expand_dims(time=times)

        zi4 = xr.DataArray(
            zi_array[:, None, None], 
            dims=("zi", "lat", "lon")
        ).expand_dims(time=times)

        ds = xr.Dataset(
            {
                "zeta": (["time", "lat", "lon"], np.full((len(times), 1, 1), 50)),
                "nuh": (["time", "zi", "lat", "lon"], K[:, :, np.newaxis, np.newaxis])
            },
            coords={
                "time": times,
                "z":    z4,     # now 4D → auxiliary coord
                "zi":   zi4,    # now 4D → auxiliary coord
                "lat":  ("lat", [50.25]),
                "lon":  ("lon", [-4.217]),
            },
            attrs={
                "description": "Synthetic turbulent diffusivity data (GOTM convention)",
                "z_units":   "m (height above sea bed, cell centres)",
                "zi_units":  "m (height above sea bed, layer interfaces)",
                "nuh_units": "m^2/s",
            }
        )
        return ds

    # ------------------------------------------------------------------
    def create_interpolated_dataset(self, ds_fine, dt_coarse_seconds):
        """
        Produce a coarser-in-time dataset by sub-sampling ds_fine.

        We take every N-th time point rather than re-computing from scratch,
        so the coarse and fine datasets share the same underlying physics —
        the only difference is temporal resolution.

        Parameters
        ----------
        ds_fine           : fine-resolution xarray Dataset (output of create_dataset)
        dt_coarse_seconds : desired coarse time step in seconds
        """
        dt_fine = float(
            (pd.Timestamp(ds_fine.time.values[1]) -
             pd.Timestamp(ds_fine.time.values[0])).total_seconds()
        )
        stride = max(1, int(round(dt_coarse_seconds / dt_fine)))
        ds_coarse = ds_fine.isel(time=slice(None, None, stride))
        return ds_coarse


# ---------------------------------------------------------------------------
# PyLag simulation wrapper
# ---------------------------------------------------------------------------

class PyLagSimulator:
    """
    Prepare inputs and launch a single PyLag GOTM-mode simulation.

    Parameters
    ----------
    dataset    : xarray Dataset with 'nuh' on (time, z)
    n_particles: number of test particles
    dt         : PyLag integration time step in seconds
    """

    def __init__(self, dataset, n_particles=100, dt=5.0):
        self.dataset    = dataset
        self.n_particles = n_particles
        self.dt         = dt

    # ------------------------------------------------------------------
    def _create_config(self, data_dir, output_dir):
        """Return a fully populated ConfigParser for this run."""
        config = ConfigParser()

        start_time = pd.Timestamp(self.dataset.time.values[0]
                                  ).strftime('%Y-%m-%d %H:%M:%S')
        end_time   = pd.Timestamp(self.dataset.time.values[-2]
                                  ).strftime('%Y-%m-%d %H:%M:%S')

        # ---- GENERAL ----
        config.add_section('GENERAL')
        config.set('GENERAL', 'log_level',   'WARNING')
        config.set('GENERAL', 'in_dir',      data_dir)
        config.set('GENERAL', 'out_dir',     output_dir)
        config.set('GENERAL', 'output_file',
                   os.path.join(output_dir, 'output', 'pylag'))

        # ---- SIMULATION ----
        config.add_section('SIMULATION')
        config.set('SIMULATION', 'simulation_type',       'trace')
        config.set('SIMULATION', 'initialisation_method', 'init_file')
        config.set('SIMULATION', 'initial_positions_file',
                   os.path.join(output_dir, 'input', 'initial_positions.dat'))
        config.set('SIMULATION', 'coordinate_system',     'cartesian')
        config.set('SIMULATION', 'depth_coordinates',     'height_above_bottom')
        config.set('SIMULATION', 'depth_restoring',     'False')
        config.set('SIMULATION', 'start_datetime',        start_time)
        config.set('SIMULATION', 'end_datetime',          end_time)
        config.set('SIMULATION', 'time_direction',        'forward')
        config.set('SIMULATION', 'number_of_particle_releases', '1')
        config.set('SIMULATION', 'particle_release_interval_in_hours', '0.0')
        config.set('SIMULATION', 'output_frequency', '3600.0')
        config.set('SIMULATION', 'sync_frequency',   '3600.0')

        # ---- RESTART ----
        config.add_section('RESTART')
        config.set('RESTART', 'restart_file_name', './restart.nc')
        config.set('RESTART', 'create_restarts', 'False')
        config.set('RESTART', 'restart_dir', './restart')
        config.set('RESTART', 'restart_frequency', '3600.0')

        # ---- NUMERICS ----
        config.add_section('NUMERICS')
        config.set('NUMERICS', 'num_method',        'standard')
        config.set('NUMERICS', 'iterative_method',  'Diff_Milstein_1D')
        config.set('NUMERICS', 'time_step_diff',    str(self.dt))

        # ---- BOUNDARY_CONDITIONS ----
        config.add_section('BOUNDARY_CONDITIONS')
        config.set('BOUNDARY_CONDITIONS', 'horiz_bound_cond', 'None')
        config.set('BOUNDARY_CONDITIONS', 'vert_bound_cond', 'reflecting')

        # ---- OCEAN_DATA ----
        config.add_section('OCEAN_DATA')
        config.set('OCEAN_DATA', 'name',            'GOTM')
        config.set('OCEAN_DATA', 'data_dir',        data_dir)
        config.set('OCEAN_DATA', 'data_file_stem',  'synthetic_data')
        config.set('OCEAN_DATA', 'rounding_interval',  '1800')
        config.set('OCEAN_DATA', 'horizontal_eddy_diffusivity_constant',  '10.0')
        config.set('OCEAN_DATA', 'grid_metrics_file',
                   os.path.join(data_dir, 'synthetic_data.nc'))
        config.set('OCEAN_DATA', 'time_dim_name',   'time')
        config.set('OCEAN_DATA', 'depth_dim_name',  'zi')
        config.set('OCEAN_DATA', 'time_var_name',   'time')
        config.set('OCEAN_DATA', 'Kz_method',       'file')
        config.set('OCEAN_DATA', 'Kz_var_name',     'nuh')
        config.set('OCEAN_DATA', 'vertical_interpolation_scheme', 'linear')

        # ---- OUTPUT ----
        config.add_section('OUTPUT')

        return config

    # ------------------------------------------------------------------
    def _create_initial_positions(self, output_dir):
        """Write the initial positions file.

        Particles are spread uniformly over the full depth range, mirroring
        the numeric example in the PyLag docs (uniform initial distribution
        is the right choice for testing the Well-Mixed Condition).
        """
        z_min = float(self.dataset.z.values.min())
        z_max = float(self.dataset.z.values.max())

        x = [0.0] * self.n_particles
        y = [0.0] * self.n_particles
        z = list(np.linspace(z_min, z_max, self.n_particles))

        input_dir = os.path.join(output_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)
        file_path = os.path.join(input_dir, 'initial_positions.dat')

        if PYLAG_AVAILABLE:
            create_initial_positions_file_single_group(
                file_path, self.n_particles, 0, x, y, z)
        else:
            # Minimal plain-text fallback (header = particle count)
            with open(file_path, 'w') as fh:
                fh.write(f'{self.n_particles}\n')
                for xi, yi, zi in zip(x, y, z):
                    fh.write(f'0 {xi} {yi} {zi}\n')

        return file_path

    # ------------------------------------------------------------------
    def run(self, output_dir):
        """
        Write data + config to disk and launch PyLag.

        Returns an xarray Dataset of the output, or None on failure.
        """
        # --- save driving data ---
        data_dir = os.path.join(output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        data_file = os.path.join(data_dir, 'synthetic_data.nc')
        self.dataset.to_netcdf(data_file)

        # --- create output subdirectory expected by PyLag ---
        os.makedirs(os.path.join(output_dir, 'output'), exist_ok=True)

        # --- initial positions ---
        self._create_initial_positions(output_dir)

        # --- config ---
        config = self._create_config(data_dir, output_dir)
        config_file = os.path.join(output_dir, 'pylag.cfg')
        with open(config_file, 'w') as fh:
            config.write(fh)

        # --- launch ---
        try:
            result = subprocess.run(
                ['python', '-m', 'pylag.main', '-c', 'pylag.cfg'],
                cwd=output_dir,
                capture_output=True,
                timeout=600
            )
            if result.returncode != 0:
                print(f'  PyLag run failed:\n{result.stderr.decode()}')
                return None

            output_file = os.path.join(output_dir, 'output', 'pylag_1.nc')
            if os.path.exists(output_file):
                return xr.open_dataset(output_file)
            else:
                print('  PyLag produced no output file.')
                return None

        except subprocess.TimeoutExpired:
            print('  PyLag simulation timed out.')
            return None
        except Exception as exc:
            print(f'  Error running PyLag: {exc}')
            return None


# ---------------------------------------------------------------------------
# Diagnostics and visualisation
# ---------------------------------------------------------------------------

class Diagnostics:
    """
    Compare two PyLag runs driven by datasets that differ only in temporal
    resolution.

    Parameters
    ----------
    ds_fine        : fine-time-step xarray Dataset (the "reference" forcing)
    ds_coarse      : coarse-time-step xarray Dataset (the "degraded" forcing)
    output_fine    : PyLag output Dataset for the fine run  (or None)
    output_coarse  : PyLag output Dataset for the coarse run (or None)
    label_fine     : legend label for the fine dataset
    label_coarse   : legend label for the coarse dataset
    """

    def __init__(self, ds_fine, ds_coarse,
                 output_fine=None, output_coarse=None,
                 label_fine='Fine (reference)',
                 label_coarse='Coarse (sub-sampled)'):
        self.ds_fine      = ds_fine
        self.ds_coarse    = ds_coarse
        self.out_fine     = output_fine
        self.out_coarse   = output_coarse
        self.label_fine   = label_fine
        self.label_coarse = label_coarse

    # ------------------------------------------------------------------
    @staticmethod
    def _particle_concentration(output, z_bins):
        """
        Return a 2-D array (n_times, n_bins) of particle counts per depth bin.
        Returns None if the output dataset or 'z' variable is unavailable.
        """
        if output is None:
            return None
        z_var = None
        for candidate in ('zpos', 'z', 'depth'):
            if candidate in output.data_vars:
                z_var = candidate
                break
        if z_var is None:
            return None

        z_particles = output[z_var].values      # (n_times, n_particles) or (n_particles,)
        if z_particles.ndim == 1:
            z_particles = z_particles[np.newaxis, :]

        concentrations = np.array([
            np.histogram(z_particles[t, :], bins=z_bins)[0]
            for t in range(z_particles.shape[0])
        ])
        return concentrations   # (n_times, n_bins)

    # ------------------------------------------------------------------
    def plot_comparison(self, figsize=(15, 11)):
        """
        Create a 2×3 panel figure:
          Row 1 – forcing data diagnostics (mean profiles, time series at mid-depth,
                   temporal coverage)
          Row 2 – particle diagnostics (final concentration, RMSE in time,
                   grid summary)
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        zi_fine   = self.ds_fine['zi'].values
        zi_coarse = self.ds_coarse['zi'].values
        # Sanity: both should share the same depth grid
        assert np.allclose(zi_fine, zi_coarse), \
            'Depth grids differ between datasets — this test keeps depth fixed!'

        # ---- Panel (0,0): mean K(zi) profiles ----
        ax = axes[0, 0]
        K_fine_mean   = self.ds_fine['nuh'].mean('time').values
        K_coarse_mean = self.ds_coarse['nuh'].mean('time').values

        ax.plot(K_fine_mean,   zi_fine,   '-',  lw=2,   label=self.label_fine,   color='steelblue')
        ax.plot(K_coarse_mean, zi_coarse, '--', lw=1.5, label=self.label_coarse, color='tomato')
        ax.set_xlabel('Mean K (m² s⁻¹)')
        ax.set_ylabel('Height above sea bed (m)')
        ax.set_title('Mean diffusivity profile')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Panel (0,1): K time series at mid-depth interface ----
        ax = axes[0, 1]
        mid_idx = len(zi_fine) // 2

        times_fine   = pd.to_datetime(self.ds_fine['time'].values)
        times_coarse = pd.to_datetime(self.ds_coarse['time'].values)

        ax.plot(times_fine,   self.ds_fine['nuh'][:, mid_idx].values,
                '-',  lw=1.5, label=self.label_fine,   color='steelblue')
        ax.plot(times_coarse, self.ds_coarse['nuh'][:, mid_idx].values,
                'o--', lw=1.2, ms=4, label=self.label_coarse, color='tomato')
        ax.set_xlabel('Time')
        ax.set_ylabel('K (m² s⁻¹)')
        ax.set_title(f'K time series at zi = {zi_fine[mid_idx]:.1f} m')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=7)

        # ---- Panel (0,2): temporal coverage comparison ----
        ax = axes[0, 2]
        dt_fine_s   = float((times_fine[1]   - times_fine[0]).total_seconds())
        dt_coarse_s = float((times_coarse[1] - times_coarse[0]).total_seconds())
        labels = [self.label_fine, self.label_coarse]
        counts = [len(times_fine), len(times_coarse)]
        dts    = [dt_fine_s, dt_coarse_s]
        colors = ['steelblue', 'tomato']
        bars   = ax.bar(labels, counts, color=colors, alpha=0.8, width=0.5)
        for bar, dt in zip(bars, dts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'Δt = {dt:.0f} s', ha='center', va='bottom', fontsize=8)
        ax.set_ylabel('Number of time steps')
        ax.set_title('Temporal resolution')
        ax.grid(True, alpha=0.3, axis='y')

        # ---- Panel (1,0): final particle concentration ----
        ax = axes[1, 0]
        z_bins    = np.linspace(float(zi_fine.min()), float(zi_fine.max()), 21)
        z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

        conc_fine   = self._particle_concentration(self.out_fine,   z_bins)
        conc_coarse = self._particle_concentration(self.out_coarse, z_bins)

        if conc_fine is not None:
            ax.plot(conc_fine[-1, :],   z_centers, '-',  lw=2,
                    label=self.label_fine,   color='steelblue')
        if conc_coarse is not None:
            ax.plot(conc_coarse[-1, :], z_centers, '--', lw=1.5,
                    label=self.label_coarse, color='tomato')

        if conc_fine is None and conc_coarse is None:
            ax.text(0.5, 0.5, 'No particle output\n(PyLag not run)',
                    ha='center', va='center', transform=ax.transAxes,
                    color='grey', fontsize=10)
        else:
            ax.legend(fontsize=8)

        ax.set_xlabel('Particle count per bin')
        ax.set_ylabel('Height above sea bed (m)')
        ax.set_title('Final particle concentration')
        ax.grid(True, alpha=0.3)

        # ---- Panel (1,1): RMSE of concentration in time ----
        ax = axes[1, 1]
        if conc_fine is not None and conc_coarse is not None:
            # Align in time (coarse may have fewer output snapshots)
            n_common = min(conc_fine.shape[0], conc_coarse.shape[0])
            cf = conc_fine[:n_common, :]
            cc = conc_coarse[:n_common, :]

            rmse = np.sqrt(np.mean((cf - cc) ** 2, axis=1))
            # Normalise by mean particle density per bin
            ref  = np.mean(cf)
            rmse_norm = rmse / ref if ref > 0 else rmse

            t_hours = np.arange(n_common)  # proxy: output snapshot index
            ax.plot(t_hours, rmse_norm, '-o', ms=4, color='purple', lw=1.5)
            ax.set_xlabel('Output snapshot index')
            ax.set_ylabel('Normalised RMSE')
            ax.set_title('Concentration RMSE (coarse vs fine)')
        else:
            ax.text(0.5, 0.5, 'No particle output\n(PyLag not run)',
                    ha='center', va='center', transform=ax.transAxes,
                    color='grey', fontsize=10)
        ax.grid(True, alpha=0.3)

        # ---- Panel (1,2): summary table ----
        ax = axes[1, 2]
        ax.axis('off')
        rows = [
            ['Depth levels',        str(len(z_fine))],
            ['Depth range (m)',     f'{z_fine.min():.1f} – {z_fine.max():.1f}'],
            ['Fine Δt (s)',         f'{dt_fine_s:.0f}'],
            ['Coarse Δt (s)',       f'{dt_coarse_s:.0f}'],
            ['Fine time steps',     str(len(times_fine))],
            ['Coarse time steps',   str(len(times_coarse))],
            ['Particles',           str(self.out_fine['zpos'].shape[-1]
                                         if self.out_fine is not None else 'N/A')],
            ['PyLag Δt (s)',        'see config'],
        ]
        table = ax.table(cellText=rows,
                         colLabels=['Parameter', 'Value'],
                         loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.6)
        ax.set_title('Run summary', pad=12)

        fig.suptitle('PyLag temporal resolution sensitivity', fontsize=13, y=1.01)
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Test PyLag sensitivity to temporal resolution of forcing data'
    )
    parser.add_argument('--z-min',  type=float, default=0.0,
                        help='Minimum height above sea bed (m)')
    parser.add_argument('--z-max',  type=float, default=50.0,
                        help='Maximum height above sea bed / water depth (m)')
    parser.add_argument('--dz',     type=float, default=1.0,
                        help='Vertical grid spacing — same for both runs (m)')
    parser.add_argument('--duration-hours', type=float, default=24.0,
                        help='Simulation duration (hours)')
    parser.add_argument('--dt-fine',   type=float, default=1800.0,
                        help='Fine forcing time step (s) — the reference dataset')
    parser.add_argument('--dt-coarse', type=float, default=10800.0,
                        help='Coarse forcing time step (s) — the degraded dataset')
    parser.add_argument('--dt-pylag',  type=float, default=5.0,
                        help='PyLag integration time step (s)')
    parser.add_argument('--n-particles', type=int, default=100,
                        help='Number of particles')
    parser.add_argument('--output-dir',  type=str, default='/home/banga/repos/PyLag/CUSTARD_PyLag/pylag_test_output',
                        help='Root output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('PyLag Temporal Resolution Sensitivity Test')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Build depth grid (shared by both runs)
    # ------------------------------------------------------------------
    z_array = np.arange(args.z_min, args.z_max + args.dz, args.dz)
    print(f'\nDepth grid : {len(z_array)} levels, dz = {args.dz} m, '
          f'range = [{args.z_min}, {args.z_max}] m')

    # ------------------------------------------------------------------
    # Generate synthetic forcing data
    # ------------------------------------------------------------------
    print('\nGenerating synthetic K(z, t) data ...')
    gen = SyntheticDataGenerator(
        z_min=args.z_min,
        z_max=args.z_max,
        t_duration_hours=args.duration_hours
    )

    # Fine dataset — the reference
    ds_fine = gen.create_dataset(z_array, args.dt_fine)
    # Coarse dataset — sub-sampled from the fine one
    ds_coarse = gen.create_interpolated_dataset(ds_fine, args.dt_coarse)
    gotm_example = xr.load_dataset("/home/banga/repos/PyLag/CUSTARD_PyLag/examples/GOTM/data/gotm_l4_20_level.nc")

    print(f'  Fine   : {len(ds_fine.time)} time steps, Δt = {args.dt_fine:.0f} s')
    print(f'  Coarse : {len(ds_coarse.time)} time steps, '
          f'Δt ≈ {args.dt_coarse:.0f} s  (stride {len(ds_fine.time) // len(ds_coarse.time)}×)')

    # ------------------------------------------------------------------
    # Run PyLag simulations
    # ------------------------------------------------------------------
    output_fine   = None
    output_coarse = None

    if PYLAG_AVAILABLE:
        print('\nRunning PyLag simulations ...')

        fine_dir   = os.path.join(args.output_dir, 'fine')
        coarse_dir = os.path.join(args.output_dir, 'coarse')
        os.makedirs(fine_dir,   exist_ok=True)
        os.makedirs(coarse_dir, exist_ok=True)

        print('  Fine run ...')
        sim_fine = PyLagSimulator(ds_fine, args.n_particles, args.dt_pylag)
        output_fine = sim_fine.run(fine_dir)

        print('  Coarse run ...')
        sim_coarse = PyLagSimulator(ds_coarse, args.n_particles, args.dt_pylag)
        output_coarse = sim_coarse.run(coarse_dir)

        if output_fine is not None:
            print('  Fine run   : OK')
        if output_coarse is not None:
            print('  Coarse run : OK')
    else:
        print('\nPyLag not available — skipping simulation step.')

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    print('\nGenerating diagnostics ...')
    diag = Diagnostics(
        ds_fine=ds_fine,
        ds_coarse=ds_coarse,
        output_fine=output_fine,
        output_coarse=output_coarse,
        label_fine=f'Fine (Δt = {args.dt_fine:.0f} s)',
        label_coarse=f'Coarse (Δt ≈ {args.dt_coarse:.0f} s)',
    )
    fig = diag.plot_comparison()
    plot_path = os.path.join(args.output_dir, 'comparison_plot.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'  Plot saved to {plot_path}')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'Depth levels      : {len(z_array)}  (dz = {args.dz} m)')
    print(f'Fine time steps   : {len(ds_fine.time)}  (Δt = {args.dt_fine:.0f} s)')
    print(f'Coarse time steps : {len(ds_coarse.time)}  (Δt ≈ {args.dt_coarse:.0f} s)')
    print(f'Particles         : {args.n_particles}')
    print(f'PyLag Δt          : {args.dt_pylag} s')
    print(f'Duration          : {args.duration_hours} h')
    if output_fine is not None and output_coarse is not None:
        print('\nBoth simulations completed successfully.')
    elif not PYLAG_AVAILABLE:
        print('\nNote: simulations skipped (PyLag not installed).')
    else:
        print('\nNote: one or both simulations did not complete.')
    print(f'\nOutputs in : {args.output_dir}')


if __name__ == '__main__':
    main()