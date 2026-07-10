"""
models/film_unet_gan.py
------------------------------
FiLM-conditioned, resolution-adaptive downscaling generator (U-Net family:
encoder/decoder with skip connections, FiLM applied at every level).

Design for arbitrary/dynamic resolution:
  - The coarse dynamic input is bilinearly upsampled to the static
    context's resolution FIRST (read from static_in.shape[-2:] at
    forward-time), then concatenated with the static context -- this turns
    the coarse-to-fine problem into a same-resolution image-to-image
    problem, which is what a U-Net's skip connections need.
  - A U-Net's downsampling path needs spatial dims divisible by
    2^depth for skip connections to align exactly. Real target grids
    (e.g. 128x128) may not always divide cleanly, and this must not
    silently break on odd sizes. Fix: pad up to the next multiple of
    2^depth (reflect padding) before the encoder, crop back to the exact
    original size after the decoder. This makes the whole network robust
    to ANY input resolution, not just convenient powers of 2.

Conditioning vector layout (shape [B, 3]):
    [0] lead_time_norm  -- lead time in days / max_lead_days
    [1] sin_doy         -- sin(2*pi * day_of_year / 365)
    [2] cos_doy         -- cos(2*pi * day_of_year / 365)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class FiLMConvBlock(nn.Module):
    """Conv -> BN -> FiLM -> activation, used as the basic unit in both
    the encoder and decoder paths."""
    def __init__(self, in_ch, out_ch, cond_dim, activation="relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.film = FiLM(cond_dim, out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True) if activation == "leaky" else nn.ReLU(inplace=True)

    def forward(self, x, cond):
        return self.act(self.film(self.bn(self.conv(x)), cond))


class FiLMDownBlock(nn.Module):
    """Two conv blocks then a stride-2 downsample. Returns (skip, downsampled)."""
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.block1 = FiLMConvBlock(in_ch, out_ch, cond_dim, activation="leaky")
        self.block2 = FiLMConvBlock(out_ch, out_ch, cond_dim, activation="leaky")
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x, cond):
        x = self.block1(x, cond)
        skip = self.block2(x, cond)
        down = self.downsample(skip)
        return skip, down


class FiLMUpBlock(nn.Module):
    """Upsample, concat with skip connection, then two conv blocks."""
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.block1 = FiLMConvBlock(out_ch + skip_ch, out_ch, cond_dim)
        self.block2 = FiLMConvBlock(out_ch, out_ch, cond_dim)

    def forward(self, x, skip, cond):
        x = self.upsample(x)
        # Guard against off-by-one size mismatches from odd input dims
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        return x


class FiLMUNetGenerator(nn.Module):
    def __init__(self, dynamic_in_channels, static_in_channels, cond_dim=3,
                 base_channels=64, depth=3, out_channels=1):
        super().__init__()
        self.depth = depth
        in_ch = dynamic_in_channels + static_in_channels

        self.stem = FiLMConvBlock(in_ch, base_channels, cond_dim, activation="leaky")

        # Encoder -- track the exact output channel count at each level so
        # the decoder can be built from real numbers, not a formula guess.
        self.down_blocks = nn.ModuleList()
        encoder_channels = []
        ch = base_channels
        for _ in range(depth):
            out_ch = ch * 2
            self.down_blocks.append(FiLMDownBlock(ch, out_ch, cond_dim))
            encoder_channels.append(out_ch)
            ch = out_ch

        # Bottleneck operates at the deepest encoder channel count
        self.bottleneck1 = FiLMConvBlock(ch, ch, cond_dim, activation="leaky")
        self.bottleneck2 = FiLMConvBlock(ch, ch, cond_dim, activation="leaky")

        # Decoder: mirror the encoder exactly using its real channel counts,
        # deepest skip first. Each level's output channel count is the NEXT
        # shallower skip's channel count (or base_channels for the last one,
        # since there's no shallower skip left to match).
        self.up_blocks = nn.ModuleList()
        running_ch = ch
        for i in reversed(range(depth)):
            skip_ch = encoder_channels[i]
            out_ch = encoder_channels[i - 1] if i > 0 else base_channels
            self.up_blocks.append(FiLMUpBlock(running_ch, skip_ch, out_ch, cond_dim))
            running_ch = out_ch

        self.final = nn.Sequential(
            nn.Conv2d(running_ch, out_channels, kernel_size=1),
            nn.ReLU(),  # non-negativity constraint for physical precipitation
        )

    def forward(self, dynamic_in, static_in, cond):
        """
        dynamic_in : [B, C_dyn, H_coarse, W_coarse]
        static_in  : [B, C_static, H_fine, W_fine]  -- defines output
                     resolution dynamically
        cond       : [B, 3] -- [lead_time_norm, sin_doy, cos_doy]
        """
        target_size = static_in.shape[-2:]

        # Bring the coarse dynamic input up to the target resolution FIRST,
        # so the U-Net operates as a same-resolution image-to-image network.
        dynamic_upsampled = F.interpolate(dynamic_in, size=target_size, mode="bilinear", align_corners=False)
        x = torch.cat([dynamic_upsampled, static_in], dim=1)

        # Pad up to a multiple of 2^depth so downsampling/upsampling align
        # exactly regardless of the actual target resolution's divisibility.
        h, w = x.shape[-2:]
        factor = 2 ** self.depth
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = self.stem(x, cond)

        skips = []
        for down_block in self.down_blocks:
            skip, x = down_block(x, cond)
            skips.append(skip)

        x = self.bottleneck1(x, cond)
        x = self.bottleneck2(x, cond)

        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip, cond)

        out = self.final(x)

        # Crop back to the exact requested target size (removes any padding)
        out = out[:, :, :target_size[0], :target_size[1]]
        return out
