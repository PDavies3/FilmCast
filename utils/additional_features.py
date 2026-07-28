"""
utils/additional_features.py
-------------------------------
Registry of COMPUTED additional-data features -- ones with no file behind
them at all, generated on the fly from sample context (valid date, the
region's actual lat/lon grid). Contrast with FILE-BACKED features
(elevation, land_mask, ...), which just need a path in
`additional_data_paths` and involve no code here.

To add a new COMPUTED feature: write a function taking `ctx` (a dict with
whatever context it needs) and returning a numpy array of shape
[n_channels, H, W], then register it in COMPUTED_FEATURES. Everything else
(config wiring, concatenation into static_input) is generic and doesn't
change.

ctx contains:
    valid_time : numpy.datetime64 -- this sample's actual forecast-valid date
    shape       : (H, W) -- the target spatial shape to broadcast into
    lat         : 1D array of latitude values at the target resolution (len H)
    lon         : 1D array of longitude values at the target resolution (len W)
"""
import numpy as np


def compute_day_of_year(ctx):
    """2 channels: sin/cos of day-of-year, broadcast as constant fields
    across the whole spatial grid. Cyclic encoding (not raw day number) so
    Dec 31 and Jan 1 are correctly adjacent rather than maximally far
    apart -- same reasoning as utils/conditioning.py's cond vector, this
    is the SPATIAL-CHANNEL version of the same information for models that
    want it as an input feature rather than (or in addition to) a FiLM
    conditioning vector."""
    vt = np.datetime64(ctx["valid_time"])
    doy = (vt - vt.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int) + 1
    angle = 2 * np.pi * doy / 365.0
    sin_channel = np.full(ctx["shape"], np.sin(angle), dtype=np.float32)
    cos_channel = np.full(ctx["shape"], np.cos(angle), dtype=np.float32)
    return np.stack([sin_channel, cos_channel], axis=0)


def compute_latlon(ctx):
    """2 channels: normalized latitude and longitude value at every pixel
    (not just a bounding-box constant), giving the model direct spatial
    position awareness. Normalized to roughly [-1, 1] (lat/90, lon/180)
    since raw degree values would sit on a very different scale than the
    zscore-normalized physical variables."""
    lon_grid, lat_grid = np.meshgrid(ctx["lon"], ctx["lat"])
    lat_channel = (lat_grid / 90.0).astype(np.float32)
    lon_channel = (lon_grid / 180.0).astype(np.float32)
    return np.stack([lat_channel, lon_channel], axis=0)


COMPUTED_FEATURES = {
    "day_of_year": compute_day_of_year,
    "latlon": compute_latlon,
}
