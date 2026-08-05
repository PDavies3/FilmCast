"""
models/critic_patchgan.py
----------------------------
Conditional PatchGAN CRITIC for WGAN-GP training (see scripts/train_wgan_gp.py).
Adapted from "The GAN Book" (Kartik Chaudhary), Skill 7 -- WGAN-GP -- and
Skill 8 -- Pix2Pix's PatchGAN discriminator design -- combined for this
repo's conditional, paired image-to-image downscaling task.

Two deliberate differences from the existing models/discriminator.py:

1. CONDITIONED ON THE PAIRED INPUT, not just the field being judged.
   The existing Pix2PixDiscriminator only ever sees the 1-channel
   precipitation field (real or fake) -- it has no way to know what
   dynamic/static context that field is supposed to correspond to, so it
   can only judge "does this look like a plausible precip field in
   general", not "does this look right GIVEN today's inputs". A real
   pix2pix-style discriminator is conditioned on the input pair -- this
   one concatenates [field, dynamic_context_upsampled_to_target_res,
   static_context] before the first conv, same idea as concatenating
   source+target images in the original Pix2Pix paper.

2. NO final sigmoid, NO spectral norm. WGAN-GP's critic outputs an
   unbounded realness score (trained on the Wasserstein distance, not a
   real/fake probability) -- a sigmoid would clip that. The Lipschitz
   constraint WGAN needs is enforced by the gradient penalty term in
   scripts/train_wgan_gp.py instead, which is the whole point of
   "gradient penalty" replacing weight clipping (and, by extension,
   replacing the need for spectral norm too -- both are alternative ways
   of enforcing the same constraint; combining them is redundant).

GroupNorm (not BatchNorm) is used for the same reason as the generators
in this repo: BatchNorm's running statistics mix information ACROSS the
batch, which does not compose correctly with the gradient penalty (that
term is computed per-sample, on gradients w.r.t. each individual
interpolated sample) -- this is a real, not just batch-size-related,
incompatibility for WGAN-GP specifically. GroupNorm normalizes within
each sample and has no such issue.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.norm_layers import make_group_norm


class PatchGANCritic(nn.Module):
    def __init__(self, field_channels, dynamic_channels, static_channels, base_channels=64):
        super().__init__()
        in_channels = field_channels + dynamic_channels + static_channels

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            make_group_norm(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            make_group_norm(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # PatchGAN: final layer outputs a SPATIAL MAP of per-patch
            # realness scores, not one scalar for the whole image -- each
            # output pixel judges a local receptive-field patch, which is
            # what makes this a "Patch" GAN. No activation -- unbounded
            # Wasserstein score per patch.
            nn.Conv2d(base_channels * 4, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, field, dynamic_in, static_in):
        """
        field       : [B, field_channels, H_fine, W_fine]  -- real or fake target
        dynamic_in  : [B, C_dyn, H_coarse, W_coarse]        -- coarse context
        static_in   : [B, C_static, H_fine, W_fine]         -- fine-res context
        """
        target_size = field.shape[-2:]
        dynamic_upsampled = F.interpolate(dynamic_in, size=target_size, mode="bilinear", align_corners=False)
        x = torch.cat([field, dynamic_upsampled, static_in], dim=1)
        return self.model(x)
