"""
utils/grib_dataset.py
----------------------
Dataloader for the real ECMWF S2S GRIB2 archive.

CONFIRMED real structure (from inspect_grib.py output against kit-test-data):
  - "time" (n=20) is the REFORECAST YEAR axis: same calendar day (May 1) each
    year from 2005-2024 -- NOT a single init date. Each file covers 20 years.
  - "step" is forecast lead time, but the RESOLUTION DIFFERS BY FILE:
      * surface temp file (mx2t6/mn2t6): 184 steps, 6-hourly
      * pressure-level file (u/v):        47 steps, DAILY
    A shared raw step_idx across files is therefore WRONG -- index 47 means
    a different lead time (or doesn't exist) depending on the file. This
    loader aligns on the actual lead-time VALUE instead: the coarsest
    (fewest-step / lowest-resolution) variable group defines the canonical
    daily sampling schedule, and every other variable is selected via
    nearest-match on its real step timedelta, not by raw index.
  - "number" (n=10) is the ensemble member.
  - "valid_time" is a (time, step) coordinate cfgrib computes automatically
    (= time + step) -- this is the real calendar date to align against
    IMERG, and is used directly here instead of manual date arithmetic.

Every (file_date, time_idx, step_idx, member_idx) combination is one
training sample -- step_idx here indexes into the CANONICAL step schedule,
not any one file's raw step axis.

A single GRIB file can pack multiple incompatible "hypercubes" (different
typeOfLevel/stepType) -- a plain xr.open_dataset() silently returns only
ONE of them. This loader always enumerates every hypercube via
cfgrib.open_datasets() so variables don't get silently hidden.
"""
import os
import glob
import re
import numpy as np
import torch
import xarray as xr
import cfgrib
from torch.utils.data import Dataset, DataLoader

import config_grib as cfg


def _open_all_hypercubes(path):
    """Open every hypercube in a GRIB file. indexpath='' avoids trying to
    write a .idx cache file next to the source (fails on read-only mounts
    like Kaggle's /kaggle/input)."""
    return cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})


class S2SGribDataset(Dataset):
    def __init__(self, data_root=None, split="train"):
        self.data_root = os.path.expanduser(data_root or cfg.DATA_ROOT)
        self.split = split

        self.init_dates = self._discover_init_dates()
        if not self.init_dates:
            raise FileNotFoundError(
                f"No init-date file sets found under {self.data_root} "
                f"(searched recursively). Expected files matching pattern like "
                f"'ECMWF-s2s-Forecast_leadtime_sfc_instantaneous_enfh_pf_init_YYYY-MM-DD.grib'"
            )

        # Caches: {(date, file_group): [list of hypercube xr.Datasets]}
        self._hypercube_cache = {}
        # Caches: {(date, file_group, shortname): xr.DataArray}
        self._var_cache = {}
        self._static_ds = None
        self._imerg_cache = {}

        # --- Determine the CANONICAL step schedule ---
        # Scan every configured variable group's step axis on the first
        # date, and use the one with the FEWEST steps (coarsest resolution)
        # as the sampling grid every other variable aligns to.
        first_date = self.init_dates[0]
        all_groups = set(cfg.SURFACE_VARIABLES.values()) | set(cfg.ATMOSPHERIC_VARIABLES.values())
        group_steps = {}
        for group in all_groups:
            var = next(v for v, g in {**cfg.SURFACE_VARIABLES, **cfg.ATMOSPHERIC_VARIABLES}.items() if g == group)
            da = self._get_var(first_date, group, var)
            group_steps[group] = da[cfg.LEADTIME_DIM].values

        coarsest_group = min(group_steps, key=lambda g: len(group_steps[g]))
        self.canonical_steps = group_steps[coarsest_group]
        self.n_steps = len(self.canonical_steps)
        print(f"[S2SGribDataset] Using '{coarsest_group}' as canonical step "
              f"schedule ({self.n_steps} steps). Other variables are matched "
              f"to these lead times via nearest-neighbor.")

        probe_var, probe_group = next(iter(cfg.SURFACE_VARIABLES.items()))
        probe_da = self._get_var(first_date, probe_group, probe_var)
        self.n_time = probe_da.sizes.get(cfg.TIME_DIM, 1)
        self.n_members = probe_da.sizes.get(cfg.MEMBER_DIM, 1)

        # Flat index over every (date, time, step, member) combination.
        # step here indexes into self.canonical_steps, not any raw file axis.
        self.index = [
            (d, t, s, m)
            for d in self.init_dates
            for t in range(self.n_time)
            for s in range(self.n_steps)
            for m in range(self.n_members)
        ]

    def _discover_init_dates(self):
        any_template = next(iter(cfg.FILENAME_TEMPLATES.values()))
        pattern = any_template.format(date="*")
        matches = glob.glob(os.path.join(self.data_root, "**", pattern), recursive=True)
        dates = []
        date_re = re.compile(r"init_(\d{4}-\d{2}-\d{2})")
        for m in matches:
            match = date_re.search(os.path.basename(m))
            if match:
                dates.append(match.group(1))
        return sorted(set(dates))

    def _resolve_path(self, date, file_group):
        filename = cfg.FILENAME_TEMPLATES[file_group].format(date=date)
        matches = glob.glob(os.path.join(self.data_root, "**", filename), recursive=True)
        if not matches:
            raise FileNotFoundError(f"Could not locate {filename} under {self.data_root}")
        return matches[0]

    def _get_hypercubes(self, date, file_group):
        key = (date, file_group)
        if key not in self._hypercube_cache:
            path = self._resolve_path(date, file_group)
            self._hypercube_cache[key] = _open_all_hypercubes(path)
        return self._hypercube_cache[key]

    def _get_var(self, date, file_group, shortname):
        key = (date, file_group, shortname)
        if key not in self._var_cache:
            hypercubes = self._get_hypercubes(date, file_group)
            found = None
            available = []
            for ds in hypercubes:
                available.extend(ds.data_vars.keys())
                if shortname in ds.data_vars:
                    found = ds[shortname]
                    break
            if found is None:
                raise KeyError(
                    f"Variable '{shortname}' not found in any hypercube of "
                    f"{cfg.FILENAME_TEMPLATES[file_group].format(date=date)}. "
                    f"Variables actually present: {sorted(set(available))}"
                )
            self._var_cache[key] = found
        return self._var_cache[key]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        date, time_idx, canon_step_idx, member_idx = self.index[idx]
        target_step = self.canonical_steps[canon_step_idx]

        # --- Surface (single-level) variables ---
        surface_layers = []
        for var, group in cfg.SURFACE_VARIABLES.items():
            da = self._get_var(date, group, var)
            sel = da.isel(**{cfg.TIME_DIM: time_idx, cfg.MEMBER_DIM: member_idx}) \
                    .sel(**{cfg.LEADTIME_DIM: target_step}, method="nearest")
            surface_layers.append(np.asarray(sel.values))

        # --- Multi-level (pressure-level) variables ---
        atmos_layers = []
        for var, group in cfg.ATMOSPHERIC_VARIABLES.items():
            da = self._get_var(date, group, var)
            base_sel = da.isel(**{cfg.TIME_DIM: time_idx, cfg.MEMBER_DIM: member_idx}) \
                         .sel(**{cfg.LEADTIME_DIM: target_step}, method="nearest")
            for level in cfg.PRESSURE_LEVELS:
                sel = base_sel.sel(isobaricInhPa=level)
                atmos_layers.append(np.asarray(sel.values))

        dynamic_tensor = torch.from_numpy(
            np.stack(surface_layers + atmos_layers, axis=0)
        ).float()

        # Real calendar date this sample corresponds to, from the CANONICAL
        # variable group (so it matches target_step exactly, not a nearest-
        # matched approximation) -- use this to align against IMERG.
        probe_var, probe_group = next(iter(cfg.SURFACE_VARIABLES.items()))
        probe_da = self._get_var(date, probe_group, probe_var)
        nearest_step = probe_da[cfg.LEADTIME_DIM].sel(
            **{cfg.LEADTIME_DIM: target_step}, method="nearest"
        ).values
        valid_time = probe_da["valid_time"].isel(**{cfg.TIME_DIM: time_idx}) \
                                            .sel(**{cfg.LEADTIME_DIM: nearest_step}).values

        # --- Static high-resolution context (opened once, reused) ---
        if self._static_ds is None:
            static_path = self._find_static_or_imerg(cfg.STATIC_LAYERS_PATH)
            self._static_ds = xr.open_dataset(static_path)
        static_tensor = torch.from_numpy(
            np.stack([self._static_ds[v].values for v in cfg.STATIC_GEOGRAPHIC_VARIABLES], axis=0)
        ).float()

        # --- Target IMERG precipitation, aligned via valid_time ---
        year = str(np.datetime_as_string(valid_time, unit="Y"))
        if year not in self._imerg_cache:
            imerg_path = self._find_static_or_imerg(cfg.IMERG_FILENAME_TEMPLATE.format(year=year))
            self._imerg_cache[year] = xr.open_dataset(imerg_path)
        ds_target = self._imerg_cache[year]
        target_sel = ds_target[cfg.IMERG_PRECIP_VAR].sel({cfg.IMERG_TIME_VAR: valid_time}, method="nearest")
        target_tensor = torch.from_numpy(np.asarray(target_sel.values)).float().unsqueeze(0)

        return {
            "dynamic_input": dynamic_tensor,
            "static_input": static_tensor,
            "target_imerg": target_tensor,
            "meta": {"date": date, "time_idx": time_idx, "canonical_step_idx": canon_step_idx,
                     "member_idx": member_idx, "valid_time": str(valid_time)},
        }

    def _find_static_or_imerg(self, filename):
        matches = glob.glob(os.path.join(self.data_root, "**", filename), recursive=True)
        if not matches:
            raise FileNotFoundError(f"Could not locate {filename} under {self.data_root}")
        return matches[0]


def get_grib_dataloader(data_root=None, split="train", batch_size=16, num_workers=None):
    if num_workers is None:
        num_workers = min(2, os.cpu_count() or 1)
    dataset = S2SGribDataset(data_root=data_root, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
