"""
tests/conftest.py
-------------------
Self-contained fixtures -- no NetCDF/GRIB files needed for the model
tests. folder_tree_root builds a real (but tiny) mock NetCDF tree on disk
so ConfigurableDownscalingDataset can be tested against actual file I/O,
matching the CONFIRMED real layout:

    {root}/D2m_T2m_CAPE_TCW/mx2t6/number_{N}/mx2t6_{date}-{LLLL}.nc   (coarse, descending lat, ensemble)
    {root}/U_V_700_500/u/level_{700,500}/u_{date}-{LLLL}.nc           (coarse, descending lat, control-only)
    {root}/precip/control/tp_{date}-{LLLL}.nc                         (fine, ascending lat, control-only)
    {root}/static/landmask.nc
    {root}/static/elevation.nc

Both coordinate orderings are represented on purpose (coarse=descending,
fine=ascending) so region-cropping is exercised both ways, not just one.
The 'u' variable exists specifically so tests can exercise
"add a variable via config, zero code changes" against a REAL file on
disk, not just a config diff.
"""
import numpy as np
import pytest
import torch
import xarray as xr


def _write_nc(path, var_name, data, lat, lon):
    path.parent.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(data, dims=("latitude", "longitude"),
                       coords={"latitude": lat, "longitude": lon})
    xr.Dataset({var_name: da}).to_netcdf(path)


@pytest.fixture
def folder_tree_root(tmp_path):
    """
    Creates a temporary directory structure mimicking the NetCDF dataset
    tree required for testing ConfigurableDownscalingDataset.

    Returns a dict: {"root": str, "dates": [...], "lead_hours": [...], "members": [...]}
    dates deliberately span three different calendar years (2019/2020/2021)
    so date-range filtering has real boundaries to test against.
    """
    root = tmp_path / "mock_dataset"
    dates = ["2019-12-31", "2020-01-01", "2020-06-15", "2020-12-31", "2021-01-01"]
    lead_hours = [0, 24]
    members = [1, 2]
    u_levels = [700, 500]

    # Coarse grid, DESCENDING latitude (39 -> -36), matching real confirmed
    # data -- spans the "africa" preset region exactly.
    coarse_lat = np.linspace(39.0, -36.0, 10)
    coarse_lon = np.linspace(-18.0, 52.0, 12)

    # Fine grid, ASCENDING latitude, same spatial extent.
    fine_lat = np.linspace(-35.0, 37.0, 20)
    fine_lon = np.linspace(-18.0, 52.0, 24)

    rng = np.random.default_rng(0)

    for date in dates:
        for lh in lead_hours:
            for m in members:
                data = rng.normal(300.0, 5.0, size=(len(coarse_lat), len(coarse_lon))).astype("float32")
                path = root / "D2m_T2m_CAPE_TCW" / "mx2t6" / f"number_{m}" / f"mx2t6_{date}-{lh:04d}.nc"
                _write_nc(path, "mx2t6", data, coarse_lat, coarse_lon)

            # u-wind: control-only (no number_ subfolder), one file per level
            for level in u_levels:
                u_data = rng.normal(0.0, 10.0, size=(len(coarse_lat), len(coarse_lon))).astype("float32")
                u_path = root / "U_V_700_500" / "u" / f"level_{level}" / f"u_{date}-{lh:04d}.nc"
                _write_nc(u_path, "u", u_data, coarse_lat, coarse_lon)

            # target: control-only, no number_ subfolder -- one file per (date, lead_hours)
            target_data = rng.exponential(0.5, size=(len(fine_lat), len(fine_lon))).astype("float32")
            target_path = root / "precip" / "control" / f"tp_{date}-{lh:04d}.nc"
            _write_nc(target_path, "tp", target_data, fine_lat, fine_lon)

    landmask = (rng.random(size=(len(fine_lat), len(fine_lon))) > 0.5).astype("float32")
    _write_nc(root / "static" / "landmask.nc", "landmask", landmask, fine_lat, fine_lon)

    elevation = rng.normal(500.0, 300.0, size=(len(fine_lat), len(fine_lon))).astype("float32")
    _write_nc(root / "static" / "elevation.nc", "elevation", elevation, fine_lat, fine_lon)

    return {"root": str(root), "dates": dates, "lead_hours": lead_hours, "members": members}


@pytest.fixture
def sample_batch():
    torch.manual_seed(0)
    batch_size = 2
    dynamic_channels = 6
    static_channels = 2
    coarse_shape = (51, 55)
    fine_shape = (128, 128)

    return {
        "dynamic_input": torch.rand(batch_size, dynamic_channels, *coarse_shape),
        "static_input": torch.rand(batch_size, static_channels, *fine_shape),
        "target_imerg": torch.rand(batch_size, 1, *fine_shape),
    }
