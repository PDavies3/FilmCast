"""
utils/folder_dataset.py
--------------------------
Config-driven dataset for the per-variable-folder, per-(date,leadtime)-file
data layout produced by convert_grib_to_netcdf.py --split_by_day:

    {data_root}/{variable_path}/[number_{N}/][level_{L}/]{var}_{YYYY-MM-DD}-{LLLL}.nc

To add a new predictor variable: add one entry to the config's `inputs`
list (pointing at its folder). Nothing in this file needs to change.
To train on a different region: change `region` in the config -- either a
named preset ("africa", "west_africa", "ghana") or explicit bounds. Nothing
in this file needs to change either.
To restrict the samples used to a date range: set `start_date` and/or
`end_date` in the config (inclusive, "YYYY-MM-DD" strings).
To convert a variable's raw units before normalization (e.g. ERA5 tp is
in meters, target IMERG is in mm/day -- set `scale: 1000` on that input
or target entry to convert meters -> mm before normalize() ever sees it).
Omit `scale` for variables that need no conversion.
To add a new ADDITIONAL DATA feature: if it's file-backed (like elevation
or a land mask), add its name to `additional_data` and its path to
`additional_data_paths` in the config -- zero code changes. If it's
COMPUTED with no file behind it (like day_of_year or latlon), register a
function in utils/additional_features.py once.

RESOLUTION: any input, the target, or a file-backed additional_data entry
can optionally specify `resolution_deg` -- when given, that variable is
resampled (bilinear) onto a regular grid at exactly that resolution
spanning the region, regardless of what resolution the source file
actually ships at.

Each ensemble member is a separate sample.
"""
import glob
import os
import re

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from utils.additional_features import COMPUTED_FEATURES
from utils.conditioning import build_cond_vector
from utils.normalisation import normalize
from utils.regions import resolve_region

_FILENAME_RE = re.compile(r".*_(\d{4}-\d{2}-\d{2})(?:-(\d{4}))?\.nc$")
_MEMBER_RE = re.compile(r"number_(\d+)")
_LEVEL_RE = re.compile(r"level_(-?\d+)")


def _crop_region(da, region):
    """Crops a DataArray to a lat/lon bounding box, correctly handling
    BOTH ascending and descending coordinate ordering."""
    if region is None:
        return da

    lat_vals = da["latitude"].values
    lon_vals = da["longitude"].values

    lat_slice = (slice(region["lat_max"], region["lat_min"])
                 if lat_vals[0] > lat_vals[-1]
                 else slice(region["lat_min"], region["lat_max"]))
    lon_slice = (slice(region["lon_max"], region["lon_min"])
                 if lon_vals[0] > lon_vals[-1]
                 else slice(region["lon_min"], region["lon_max"]))

    cropped = da.sel(latitude=lat_slice, longitude=lon_slice)
    if cropped.sizes.get("latitude", 0) == 0 or cropped.sizes.get("longitude", 0) == 0:
        raise ValueError(
            f"Region crop produced an EMPTY array. Requested region={region}, "
            f"but this file's actual lat range is "
            f"[{lat_vals.min():.2f}, {lat_vals.max():.2f}] and lon range is "
            f"[{lon_vals.min():.2f}, {lon_vals.max():.2f}]. Check for overlap."
        )
    return cropped


def _resample_to_resolution(da, region, resolution_deg):
    """Resamples (bilinear) onto a regular grid at exactly resolution_deg
    spanning the region, regardless of the source file's native resolution."""
    if resolution_deg is None:
        return da

    lat_vals = da["latitude"].values
    lon_vals = da["longitude"].values

    n_lat = round((region["lat_max"] - region["lat_min"]) / resolution_deg) + 1
    n_lon = round((region["lon_max"] - region["lon_min"]) / resolution_deg) + 1

    target_lat = (np.linspace(region["lat_max"], region["lat_min"], n_lat)
                  if lat_vals[0] > lat_vals[-1]
                  else np.linspace(region["lat_min"], region["lat_max"], n_lat))
    target_lon = (np.linspace(region["lon_max"], region["lon_min"], n_lon)
                  if lon_vals[0] > lon_vals[-1]
                  else np.linspace(region["lon_min"], region["lon_max"], n_lon))

    resampled = da.interp(latitude=target_lat, longitude=target_lon, method="linear")
    if np.any(np.isnan(resampled.values)):
        raise ValueError(
            f"Resampling to resolution_deg={resolution_deg} over region={region} "
            f"produced NaN values -- the target grid likely extends slightly "
            f"beyond the source file's actual coverage "
            f"(source lat range [{lat_vals.min():.3f}, {lat_vals.max():.3f}], "
            f"lon range [{lon_vals.min():.3f}, {lon_vals.max():.3f}]). Try "
            f"shrinking the region slightly or check the source file's extent."
        )
    return resampled


def _parse_path(path):
    """Extracts (date, lead_hours, member, level) from a file's full path."""
    fname = os.path.basename(path)
    m = _FILENAME_RE.match(fname)
    if not m:
        return None
    date_str, llll = m.groups()
    member_match = _MEMBER_RE.search(path)
    level_match = _LEVEL_RE.search(path)
    member = int(member_match.group(1)) if member_match else None
    level = int(level_match.group(1)) if level_match else None
    lead_hours = int(llll) if llll is not None else 0
    return date_str, lead_hours, member, level


class ConfigurableDownscalingDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.data_root = os.path.expanduser(config["data_root"])
        self.region = resolve_region(config.get("region"))
        self.inputs_spec = config["inputs"]
        self.target_spec = config["target"]
        self.additional_data = config.get("additional_data", [])
        self.additional_data_paths = config.get("additional_data_paths", {})
        if not self.additional_data:
            raise ValueError(
                "config['additional_data'] must have at least one entry. "
                "The FiLM generators derive their OUTPUT RESOLUTION from "
                "static_input.shape at forward-time -- with none, there's "
                "no way to know what resolution to produce."
            )
        for name in self.additional_data:
            if name not in COMPUTED_FEATURES and name not in self.additional_data_paths:
                raise ValueError(
                    f"additional_data entry '{name}' is not a registered "
                    f"computed feature ({list(COMPUTED_FEATURES.keys())}) "
                    f"and has no path in additional_data_paths."
                )
        self.max_lead_days = config.get("max_lead_days", 46.0)

        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        for label, value in (("start_date", self.start_date), ("end_date", self.end_date)):
            if value is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
                raise ValueError(
                    f"config['{label}'] = {value!r} is not in 'YYYY-MM-DD' format."
                )

        self._file_index = {}
        for spec in self.inputs_spec + [self.target_spec]:
            self._index_variable(spec)

        probe_name = self.inputs_spec[0]["name"]
        keys = [k for k in self._file_index if k[0] == probe_name]
        self.samples = sorted({(k[1], k[2], k[3]) for k in keys})

        if not self.samples:
            raise FileNotFoundError(
                f"No samples discovered for '{probe_name}' under "
                f"{os.path.join(self.data_root, self.inputs_spec[0]['path'])}."
            )

        if self.start_date is not None or self.end_date is not None:
            n_before = len(self.samples)
            self.samples = [
                s for s in self.samples
                if (self.start_date is None or s[0] >= self.start_date)
                and (self.end_date is None or s[0] <= self.end_date)
            ]
            if not self.samples:
                discovered_dates = sorted({d for d, _, _ in
                                            sorted({(k[1], k[2], k[3]) for k in keys})})
                raise ValueError(
                    f"start_date/end_date filter [{self.start_date}, {self.end_date}] "
                    f"removed all {n_before} discovered samples. Discovered dates range "
                    f"from {discovered_dates[0]} to {discovered_dates[-1]}."
                )
            print(f"[ConfigurableDownscalingDataset] Period filter [{self.start_date or '-inf'}, "
                  f"{self.end_date or '+inf'}]: {n_before} -> {len(self.samples)} samples.")

    def _index_variable(self, spec):
        var_dir = os.path.join(self.data_root, spec["path"])
        matches = glob.glob(os.path.join(var_dir, "**", "*.nc"), recursive=True)
        for path in matches:
            parsed = _parse_path(path)
            if parsed is None:
                continue
            date_str, lead_hours, member, level = parsed
            self._file_index[(spec["name"], date_str, lead_hours, member, level)] = path

    def _resolve_path(self, var_name, date_str, lead_hours, member, level):
        key = (var_name, date_str, lead_hours, member, level)
        if key in self._file_index:
            return self._file_index[key]
        fallback_key = (var_name, date_str, lead_hours, None, level)
        if fallback_key in self._file_index:
            return self._file_index[fallback_key]
        raise KeyError(
            f"No file found for variable='{var_name}', date={date_str}, "
            f"lead_hours={lead_hours}, member={member}, level={level}"
        )

    def _load_cropped(self, path, resolution_deg=None):
        ds = xr.open_dataset(path)
        var_name = list(ds.data_vars)[0]
        da_full = ds[var_name]
        if resolution_deg is not None:
            return _resample_to_resolution(da_full, self.region, resolution_deg)
        return _crop_region(da_full, self.region)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        date_str, lead_hours, member = self.samples[idx]

        # --- Dynamic inputs (coarse resolution) ---
        channels = []
        for spec in self.inputs_spec:
            levels = spec.get("levels", [None])
            for level in levels:
                path = self._resolve_path(spec["name"], date_str, lead_hours, member, level)
                da = self._load_cropped(path, spec.get("resolution_deg"))
                arr = da.values
                scale = spec.get("scale")
                if scale is not None:
                    arr = arr * scale
                arr = normalize(arr, spec["normalisation"])
                channels.append(arr)
        dynamic_tensor = torch.from_numpy(np.stack(channels, axis=0)).float()

        # --- Target (fine resolution) ---
        target_path = self._resolve_path(self.target_spec["name"], date_str, lead_hours, member, None)
        target_da = self._load_cropped(target_path, self.target_spec.get("resolution_deg"))
        target_arr = target_da.values
        target_scale = self.target_spec.get("scale")
        if target_scale is not None:
            target_arr = target_arr * target_scale
        target_arr = normalize(target_arr, self.target_spec["normalisation"])
        target_tensor = torch.from_numpy(target_arr).float().unsqueeze(0)

        # --- valid_time ---
        init_date = np.datetime64(date_str)
        valid_time = init_date + np.timedelta64(lead_hours, "h")

        # --- Additional data (fine resolution, same grid as target) ---
        ctx = {
            "valid_time": valid_time,
            "shape": target_da.shape,
            "lat": target_da["latitude"].values,
            "lon": target_da["longitude"].values,
        }
        additional_channels = []
        for name in self.additional_data:
            if name in COMPUTED_FEATURES:
                additional_channels.append(COMPUTED_FEATURES[name](ctx))
            else:
                entry = self.additional_data_paths[name]
                if isinstance(entry, dict):
                    path = os.path.join(self.data_root, entry["path"])
                    resolution_deg = entry.get("resolution_deg")
                else:
                    path = os.path.join(self.data_root, entry)
                    resolution_deg = None
                da = self._load_cropped(path, resolution_deg)
                additional_channels.append(da.values[np.newaxis, ...])
        static_tensor = torch.from_numpy(np.concatenate(additional_channels, axis=0)).float()

        cond = build_cond_vector([valid_time], [lead_hours / 24.0], self.max_lead_days)[0]

        return {
            "dynamic_input": dynamic_tensor,
            "static_input": static_tensor,
            "target": target_tensor,
            "cond": cond,
            "lat": torch.from_numpy(np.asarray(ctx["lat"]).copy()).float(),
            "lon": torch.from_numpy(np.asarray(ctx["lon"]).copy()).float(),
            "meta": {"date": date_str, "lead_hours": lead_hours,
                     "member": member if member is not None else -1},
        }
