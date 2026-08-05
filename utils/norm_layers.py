"""
utils/norm_layers.py
-----------------------
make_group_norm: GroupNorm replacement for BatchNorm2d in the FiLM
generators. BatchNorm2d's running statistics are computed over the batch
dimension -- unreliable/noisy at small batch sizes (this repo trains with
batch_size=4 on CPU), a known contributor to GAN training instability.
GroupNorm normalizes within each sample (across channel groups), so it's
completely unaffected by batch size.

num_groups is chosen as the largest divisor of `channels` that is <= 8
(8 is the common default in GroupNorm literature) -- picked dynamically
rather than hardcoded so this keeps working if base_channels is ever
changed from the current default of 64.
"""
import torch.nn as nn


def make_group_norm(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)
