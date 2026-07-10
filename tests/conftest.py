"""
tests/conftest.py
-------------------
Self-contained fixtures -- no NetCDF/GRIB files needed. This repo's test
suite is about the model architectures and conditioning logic, so fixtures
are synthetic tensors shaped like the real data (51x55 ECMWF-ish coarse
grid, 128x128 IMERG-ish target grid) rather than a full file-based pipeline.
"""
import pytest
import torch


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
