"""
utils/conditioning.py
-----------------------
Builds the [lead_time_norm, sin_doy, cos_doy] conditioning vector consumed
by the FiLM-based generators (variants 5 and 6).

Uses valid_time (= init_time + lead_time), NOT the init date, for the
day-of-year component -- lead times run out to 46 days, so a sample can
validate in a meaningfully different season than when the forecast was
issued.
"""
import numpy as np
import torch


def build_cond_vector(valid_times, lead_time_days, max_lead_days):
    """
    valid_times     : iterable of numpy.datetime64 (or ISO strings), length B
    lead_time_days   : iterable of floats, length B -- lead time in days
    max_lead_days    : float -- normalization constant (e.g. 46.0)

    Returns: torch.FloatTensor of shape [B, 3]
    """
    rows = []
    for vt, lead_days in zip(valid_times, lead_time_days):
        vt = np.datetime64(vt)
        day_of_year = (vt - vt.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int) + 1
        angle = 2 * np.pi * day_of_year / 365.0
        lead_time_norm = float(lead_days) / float(max_lead_days)
        rows.append([lead_time_norm, np.sin(angle), np.cos(angle)])
    return torch.tensor(rows, dtype=torch.float32)
