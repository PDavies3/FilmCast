"""
config_grib.py
--------------
Configuration for the REAL ECMWF S2S GRIB2 data.

CONFIRMED so far from inspect_grib.py against:
  - sfc_d2m_t2m_CAPE_TCW file -> variables found: mx2t6, mn2t6
    (NOTE: this file name suggests d2m/t2m/CAPE/TCW should also be in
    there. inspect_grib.py has been fixed to enumerate ALL hypercubes in
    a file -- re-run it against this file with the fixed version to
    confirm whether those variables exist in a different hypercube, or
    whether this file genuinely only contains mx2t6/mn2t6.)
  - u_700_500 file -> variables found: u, v (both!), at isobaricInhPa [700, 500]

STILL UNCONFIRMED (never inspected yet):
  - sfc_instantaneous file contents
  - sfc_T_max file contents
  - w_700_500 file contents (filename implies "w", but confirm)
Run inspect_grib.py against these three before trusting the entries below.
"""
import os

# DATA_ROOT resolution order:
#   1. DOWNSCALING_DATA_ROOT environment variable
#   2. --data_dir CLI flag / explicit data_root= argument (overrides at runtime)
#   3. Fallback default: ./data relative to wherever the script is run
DATA_ROOT = os.path.expanduser(
    os.environ.get("DOWNSCALING_DATA_ROOT", os.path.join(os.getcwd(), "data"))
)

# Filename template -- {date} gets substituted with e.g. "2025-05-01"
FILENAME_TEMPLATES = {
    "sfc_d2m_t2m_cape_tcw": "ECMWF-s2s-Forecast_leadtime_sfc_d2m_t2m_CAPE_TCW_enfh_pf_init_{date}.grib",
    "sfc_instantaneous":    "ECMWF-s2s-Forecast_leadtime_sfc_instantaneous_enfh_pf_init_{date}.grib",
    "sfc_t_max":            "ECMWF-s2s-Forecast_leadtime_sfc_T_max_enfh_pf_init_{date}.grib",
    "u_700_500":            "ECMWF-s2s-Forecast_leadtime_u_700_500_enfh_pf_init_{date}.grib",
    "w_700_500":            "ECMWF-s2s-Forecast_leadtime_w_700_500_enfh_pf_init_{date}.grib",
}

# Surface (single-level) variables -- CONFIRMED entries have a real
# shortName from inspect_grib.py output; UNCONFIRMED ones are still guesses.
SURFACE_VARIABLES = {
    "mx2t6": "sfc_d2m_t2m_cape_tcw",   # CONFIRMED: max 2m temp (6-hourly)
    "mn2t6": "sfc_d2m_t2m_cape_tcw",   # CONFIRMED: min 2m temp (6-hourly)
    # -- below this line: UNCONFIRMED, re-inspect before training --
    "msl":  "sfc_instantaneous",       # UNCONFIRMED guess: mean sea level pressure
    "u10":  "sfc_instantaneous",       # UNCONFIRMED guess: 10m U wind
    "v10":  "sfc_instantaneous",       # UNCONFIRMED guess: 10m V wind
    "mx2t": "sfc_t_max",                # UNCONFIRMED guess: daily max 2m temp
}

# Multi-level (pressure-level) variables. Each expands into
# len(PRESSURE_LEVELS) channels automatically.
ATMOSPHERIC_VARIABLES = {
    "u": "u_700_500",   # CONFIRMED: U wind component, isobaricInhPa [700, 500]
    "v": "u_700_500",   # CONFIRMED: V wind is ALSO in this file (bonus -- wasn't expected)
    "w": "w_700_500",   # UNCONFIRMED guess: vertical velocity -- re-inspect this file
}
PRESSURE_LEVELS = [700, 500]  # CONFIRMED hPa values from isobaricInhPa coord

# Dimension names as they actually appear in cfgrib output (confirmed)
TIME_DIM = "time"        # reforecast/hindcast YEAR axis (n=20, e.g. 2005-2024) -- NOT a single date
LEADTIME_DIM = "step"    # forecast lead time within each run (n=184, 6-hourly)
MEMBER_DIM = "number"    # ensemble member (n=10)

# Static high-resolution context (point this at your real landmask/topography file)
STATIC_GEOGRAPHIC_VARIABLES = ["landmask", "topography"]
STATIC_LAYERS_PATH = "static_layers.nc"

# Target precipitation (IMERG). Alignment uses the "valid_time" coordinate
# cfgrib computes automatically (time + step), NOT the raw file date --
# confirm your IMERG file/variable naming convention here.
IMERG_FILENAME_TEMPLATE = "imerg_{year}.nc"   # UNCONFIRMED -- adjust to your real IMERG layout
IMERG_TIME_VAR = "time"
IMERG_PRECIP_VAR = "precipitation"

# Grid geometry
COARSE_SHAPE = (51, 55)    # CONFIRMED from latitude(51)/longitude(55) dims -- NOT 9x9 as originally assumed!
FINE_SHAPE = (128, 128)     # confirm against your IMERG target resolution

# Ensemble handling: every member is a SEPARATE training sample (not averaged)
ENSEMBLE_MODE = "separate_samples"
