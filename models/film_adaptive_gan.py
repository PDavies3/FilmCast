"""
models/film_adaptive_gan.py
----------------------------------
FiLM-conditioned, resolution-adaptive downscaling generator
(residual-block family).

Design goals:

  - Output resolution is NOT hard-coded. It's read at forward-time from the
    static context tensor's spatial shape (static_in.shape[-2:]), so the
    same architecture works whether your target grid is 128x128, 256x256,
    or anything else -- no fixed upscale factor or crop-conv kernel size
    baked into the network.

  - Upsampling uses bilinear interpolation to the target size, which works
    for ANY input:output ratio -- including non-power-of-2 ratios like a
    real ECMWF (~1.5deg) -> IMERG (0.1deg) mapping (~15x). Interpolation
    alone would be blurry, so it's followed by convolutional refinement
    blocks at the target resolution to sharpen the result.

  - Lead time and day-of-year are injected via FiLM (Feature-wise Linear
    Modulation) at every residual block, rather than as extra broadcast
    input channels. This lets the network learn a genuinely different
    transformation at different lead times / seasons (e.g. trust the
    input more at short lead times, lean more on climatology/topography at
    long lead times) rather than that information just sitting passively
    alongside the weather variables.

Conditioning vector layout (shape [B, 3]):
    [0] lead_time_norm  -- lead time in days / max_lead_days, roughly [0, 1]
    [1] sin_doy         -- sin(2*pi * day_of_year / 365)
    [2] cos_doy         -- cos(2*pi * day_of_year / 365)
Use valid_time's day-of-year (not the init date's) since lead times here
run out to 46 days -- a sample can validate in a meaningfully different
season than when the forecast was issued.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.norm_layers import make_group_norm


class FiLM(nn.Module):
    """Produces per-channel scale/shift from a conditioning vector and
    applies them to a feature map. Modulation is (1 + gamma)*x + beta so
    that at initialization (small weights -> gamma,beta near 0) this starts
    close to an identity operation and the block can grow into using the
    conditioning rather than being destabilized by it from step 1."""
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        gamma_beta = self.to_gamma_beta(cond)          # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)        # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = make_group_norm(channels)  # P.Davies: add -- GroupNorm replaces BatchNorm2d
        self.film1 = FiLM(cond_dim, channels)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = make_group_norm(channels)  # P.Davies: add -- GroupNorm replaces BatchNorm2d
        self.film2 = FiLM(cond_dim, channels)

    def forward(self, x, cond):
        residual = x
        out = self.film1(self.bn1(self.conv1(x)), cond)
        out = self.act(out)
        out = self.film2(self.bn2(self.conv2(out)), cond)
        return residual + out


class SpatialAttentionGate(nn.Module):
    """Directs the network's focus toward local storm structures /
    topography transitions using a gated attention mechanism."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), make_group_norm(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), make_group_norm(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), make_group_norm(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class FiLMAdaptiveGenerator(nn.Module):
    def __init__(self, dynamic_in_channels, static_in_channels, cond_dim=3,
                 base_channels=64, num_film_blocks=8, out_channels=1):
        super().__init__()
        self.feat_extract = nn.Sequential(
            nn.Conv2d(dynamic_in_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.film_blocks = nn.ModuleList([
            FiLMResidualBlock(base_channels, cond_dim) for _ in range(num_film_blocks)
        ])

        # Post-interpolation refinement -- operates at whatever the target
        # spatial size turns out to be; no fixed size assumed anywhere here.
        self.refine = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        self.static_encoder = nn.Sequential(
            nn.Conv2d(static_in_channels, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, base_channels, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
        )
        self.attention_gate = SpatialAttentionGate(
            F_g=base_channels, F_l=base_channels, F_int=base_channels // 2
        )

        self.reconstruct = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1), nn.PReLU(),
            # P.Davies: add -- no final activation. Output is a NORMALIZED
            # value (compared against a normalized target that can be
            # negative -- e.g. zero precip normalizes to a negative
            # number), not physical precipitation. A ReLU here floors
            # output at 0, making every dry pixel's target unreachable and
            # killing gradient flow. Non-negativity belongs at physical-
            # units time (after denormalize()), not inside the network.
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, dynamic_in, static_in, cond):
        """
        dynamic_in : [B, C_dyn, H_coarse, W_coarse]  (e.g. 51x55 ECMWF grid)
        static_in  : [B, C_static, H_fine, W_fine]    (defines output res.
                     dynamically -- e.g. 128x128 IMERG grid, but could be
                     any size without touching this code)
        cond       : [B, 3] -- [lead_time_norm, sin_doy, cos_doy]
        """
        target_size = static_in.shape[-2:]

        x = self.feat_extract(dynamic_in)
        for block in self.film_blocks:
            x = block(x, cond)

        # Resolution-adaptive upsample: works for ANY coarse:fine ratio,
        # including non-power-of-2 ratios (real case here is ~15x).
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.refine(x)

        static_feats = self.static_encoder(static_in)
        gated_static = self.attention_gate(g=x, x=static_feats)

        combined = torch.cat([x, gated_static], dim=1)
        return self.reconstruct(combined)
