"""
tests/test_conditioning.py
-----------------------------
Verifies the [lead_time_norm, sin_doy, cos_doy] conditioning vector.
"""
import numpy as np
import torch
from utils.conditioning import build_cond_vector


def test_shape_and_dtype():
    cond = build_cond_vector([np.datetime64("2020-06-15")], [5.0], max_lead_days=46.0)
    assert cond.shape == (1, 3)
    assert cond.dtype == torch.float32


def test_lead_time_normalization_range():
    cond = build_cond_vector(
        [np.datetime64("2020-01-01"), np.datetime64("2020-01-01")],
        [0.0, 46.0],
        max_lead_days=46.0,
    )
    assert cond[0, 0].item() == 0.0
    assert cond[1, 0].item() == 1.0


def test_day_of_year_is_cyclic_not_discontinuous():
    """Dec 31 and Jan 1 are one day apart physically -- the encoding should
    reflect that, not treat them as maximally far apart like a raw day
    number (365 vs 1) would."""
    cond_dec31 = build_cond_vector([np.datetime64("2020-12-31")], [0.0], max_lead_days=46.0)
    cond_jan1 = build_cond_vector([np.datetime64("2021-01-01")], [0.0], max_lead_days=46.0)

    sin_cos_dec31 = cond_dec31[0, 1:3]
    sin_cos_jan1 = cond_jan1[0, 1:3]
    distance = torch.norm(sin_cos_dec31 - sin_cos_jan1).item()

    # If it were naively linear (day 365 vs day 1), these would be far apart.
    # Cyclically, they're adjacent -- distance should be small.
    assert distance < 0.1, f"Dec 31 / Jan 1 should be adjacent on the cycle, got distance {distance}"


def test_uses_valid_time_not_init_date():
    """Two different lead times from the same conceptual forecast should
    produce different day-of-year encodings if they land in different
    months -- this is what makes valid_time (not init date) the correct
    choice for a 46-day lead time range."""
    cond_short_lead = build_cond_vector([np.datetime64("2020-05-01")], [0.0], max_lead_days=46.0)
    cond_long_lead = build_cond_vector([np.datetime64("2020-06-16")], [46.0], max_lead_days=46.0)
    assert not torch.allclose(cond_short_lead[0, 1:3], cond_long_lead[0, 1:3])
