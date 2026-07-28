"""
tests/test_regions.py
------------------------
Tests the named region preset registry (utils/regions.py).
"""
import pytest
from utils.regions import resolve_region, REGIONS


def test_all_three_presets_exist():
    assert "africa" in REGIONS
    assert "west_africa" in REGIONS
    assert "ghana" in REGIONS


def test_named_preset_resolves_to_bounds_dict():
    resolved = resolve_region("ghana")
    assert resolved == REGIONS["ghana"]
    assert set(resolved.keys()) == {"lat_min", "lat_max", "lon_min", "lon_max"}


def test_explicit_bounds_pass_through_unchanged():
    custom = {"lat_min": 1.0, "lat_max": 2.0, "lon_min": 3.0, "lon_max": 4.0}
    assert resolve_region(custom) == custom


def test_unknown_preset_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown named region"):
        resolve_region("atlantis")


def test_ghana_nested_inside_west_africa_nested_inside_africa():
    """Sanity check the actual preset VALUES make geographic sense
    relative to each other, not just that the registry mechanism works."""
    africa = REGIONS["africa"]
    wa = REGIONS["west_africa"]
    ghana = REGIONS["ghana"]

    assert africa["lat_min"] <= wa["lat_min"] and wa["lat_max"] <= africa["lat_max"]
    assert africa["lon_min"] <= wa["lon_min"] and wa["lon_max"] <= africa["lon_max"]
    assert wa["lat_min"] <= ghana["lat_min"] and ghana["lat_max"] <= wa["lat_max"]
    assert wa["lon_min"] <= ghana["lon_min"] and ghana["lon_max"] <= wa["lon_max"]
