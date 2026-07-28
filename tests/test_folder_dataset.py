"""
tests/test_folder_dataset.py
-------------------------------
Tests ConfigurableDownscalingDataset against a synthetic folder tree that
matches the REAL confirmed layout, including descending latitude (39 ->
-36, like the real data) to specifically exercise the region-cropping
coordinate-ordering trap.
"""
import copy

import pytest

from utils.folder_dataset import ConfigurableDownscalingDataset


def _base_config(root):
    return {
        "data_root": root,
        "region": {"lat_min": -35.0, "lat_max": 37.0, "lon_min": -18.0, "lon_max": 52.0},
        "inputs": [
            {"name": "mx2t6", "path": "D2m_T2m_CAPE_TCW/mx2t6",
             "normalisation": {"type": "zscore", "stats": {"mean": 300.0, "std": 5.0}}},
        ],
        "target": {"name": "tp", "path": "precip/control",
                   "normalisation": {"type": "log1p", "stats": {"log1p_mean": 0.08, "log1p_std": 0.27}}},
        "additional_data": ["landmask", "elevation"],
        "additional_data_paths": {
            "landmask": "static/landmask.nc",
            "elevation": "static/elevation.nc",
        },
    }


def test_discovers_correct_sample_count(folder_tree_root):
    config = _base_config(folder_tree_root["root"])
    ds = ConfigurableDownscalingDataset(config)
    expected = len(folder_tree_root["dates"]) * len(folder_tree_root["lead_hours"]) * len(folder_tree_root["members"])
    assert len(ds) == expected


def test_adding_a_variable_via_config_only_changes_channel_count(folder_tree_root):
    """The core design goal: a new predictor variable is one config entry,
    zero code changes."""
    config = _base_config(folder_tree_root["root"])
    ds_before = ConfigurableDownscalingDataset(config)
    assert ds_before[0]["dynamic_input"].shape[0] == 1

    config_with_u = copy.deepcopy(config)
    config_with_u["inputs"].append({
        "name": "u", "path": "U_V_700_500/u", "levels": [700, 500],
        "normalisation": {"type": "zscore", "stats": {"mean": 0.0, "std": 10.0}},
    })
    ds_after = ConfigurableDownscalingDataset(config_with_u)
    assert ds_after[0]["dynamic_input"].shape[0] == 3  # mx2t6 + u@700 + u@500


def test_region_via_config_only_changes_spatial_shape(folder_tree_root):
    """The other core design goal: region is one config field, zero code
    changes."""
    config = _base_config(folder_tree_root["root"])
    full = ConfigurableDownscalingDataset(config)[0]

    config_small = copy.deepcopy(config)
    config_small["region"] = {"lat_min": 4.5, "lat_max": 11.5, "lon_min": -3.5, "lon_max": 1.5}
    small = ConfigurableDownscalingDataset(config_small)[0]

    assert small["dynamic_input"].shape[-2:] != full["dynamic_input"].shape[-2:]
    assert small["dynamic_input"].shape[-1] < full["dynamic_input"].shape[-1]
    assert small["dynamic_input"].shape[-2] < full["dynamic_input"].shape[-2]


def test_region_crop_correct_on_descending_latitude(folder_tree_root):
    """The fixture uses descending latitude (39 -> -36), matching the real
    confirmed data. A naive slice(lat_min, lat_max) silently returns EMPTY
    on descending coords rather than erroring -- this must not happen."""
    config = _base_config(folder_tree_root["root"])
    config["region"] = {"lat_min": 4.5, "lat_max": 11.5, "lon_min": -3.5, "lon_max": 1.5}
    ds = ConfigurableDownscalingDataset(config)
    sample = ds[0]
    assert sample["dynamic_input"].shape[-2] > 0
    assert sample["dynamic_input"].shape[-1] > 0


def test_out_of_bounds_region_raises_clear_error_not_silent_empty(folder_tree_root):
    config = _base_config(folder_tree_root["root"])
    config["region"] = {"lat_min": 200.0, "lat_max": 210.0, "lon_min": -18.0, "lon_max": 52.0}
    ds = ConfigurableDownscalingDataset(config)
    with pytest.raises(ValueError, match="EMPTY array"):
        ds[0]


def test_control_only_variable_no_ensemble_folder_resolves(folder_tree_root):
    """The target ('tp') has no number_ subfolder (control-only, no
    ensemble spread) -- must resolve via fallback, not crash."""
    config = _base_config(folder_tree_root["root"])
    ds = ConfigurableDownscalingDataset(config)
    sample = ds[0]
    # Target and static are both at the FINE (CHIRPS-like) resolution --
    # dynamic_input is coarse. They should match each other, not the
    # coarse dynamic_input.
    assert sample["target"].shape[-2:] == sample["static_input"].shape[-2:]


def test_static_fields_loaded_and_cropped_consistently(folder_tree_root):
    config = _base_config(folder_tree_root["root"])
    config["region"] = {"lat_min": 4.5, "lat_max": 11.5, "lon_min": -3.5, "lon_max": 1.5}
    ds = ConfigurableDownscalingDataset(config)
    sample = ds[0]
    assert sample["static_input"].shape[0] == 2  # landmask + elevation
    # static and target are both at the fine (CHIRPS-like) resolution
    assert sample["static_input"].shape[-2:] == sample["target"].shape[-2:]
    # dynamic_input stays at the coarse resolution -- NOT the same shape
    assert sample["static_input"].shape[-2:] != sample["dynamic_input"].shape[-2:]


def test_empty_additional_data_list_raises_clear_error(folder_tree_root):
    """Models derive output resolution from static_input.shape (built from
    additional_data) -- an empty list has no way to know what resolution
    to produce, and must fail clearly at dataset construction."""
    config = _base_config(folder_tree_root["root"])
    config["additional_data"] = []
    with pytest.raises(ValueError, match="additional_data"):
        ConfigurableDownscalingDataset(config)


def test_cond_vector_present_and_correct_shape(folder_tree_root):
    config = _base_config(folder_tree_root["root"])
    ds = ConfigurableDownscalingDataset(config)
    sample = ds[0]
    assert sample["cond"].shape == (3,)


def test_missing_data_root_raises_clear_error(folder_tree_root):
    config = _base_config("/nonexistent/path/that/does/not/exist")
    with pytest.raises(FileNotFoundError):
        ConfigurableDownscalingDataset(config)


def test_named_region_preset_usable_directly_in_config(folder_tree_root):
    """region can be a string preset ('ghana', 'west_africa', 'africa'),
    not just an explicit bounds dict."""
    config = _base_config(folder_tree_root["root"])
    config["region"] = "africa"  # matches the fixture's actual coverage
    ds = ConfigurableDownscalingDataset(config)
    sample = ds[0]
    assert sample["dynamic_input"].shape[-2] > 0
    assert sample["dynamic_input"].shape[-1] > 0


def test_resolution_deg_forces_consistent_shape_regardless_of_native_resolution(folder_tree_root):
    """The core new design goal: a variable's OWN resolution_deg
    determines its output shape, independent of whatever native
    resolution its source file happens to have."""
    config = _base_config(folder_tree_root["root"])
    config["inputs"][0]["resolution_deg"] = 1.0
    ds = ConfigurableDownscalingDataset(config)
    sample = ds[0]

    # Recompute the expected shape directly from the region + resolution_deg
    region = config["region"]
    expected_lat = round((region["lat_max"] - region["lat_min"]) / 1.0) + 1
    expected_lon = round((region["lon_max"] - region["lon_min"]) / 1.0) + 1
    assert sample["dynamic_input"].shape[-2:] == (expected_lat, expected_lon)


def test_omitting_resolution_deg_keeps_native_resolution_backward_compatible(folder_tree_root):
    """Without resolution_deg, behavior is unchanged from before this
    feature existed -- crop only, native resolution."""
    config = _base_config(folder_tree_root["root"])
    assert "resolution_deg" not in config["inputs"][0]
    ds = ConfigurableDownscalingDataset(config)
    sample = ds[0]
    assert sample["dynamic_input"].shape[-2] > 0  # just needs to work, shape is whatever's native
