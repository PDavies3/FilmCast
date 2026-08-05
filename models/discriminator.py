import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class Pix2PixDiscriminator(nn.Module):
    """PatchGAN discriminator. Spectral normalization on every conv layer
    constrains each layer's Lipschitz constant -- a standard GAN
    stabilization technique that directly targets the discriminator
    becoming overconfident too fast (the Loss_D -> ~0 collapse observed
    in training). InstanceNorm2d replaces BatchNorm2d -- it normalizes
    per-sample rather than across the batch, so it's unaffected by the
    small batch_size=4 used here."""
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)),
        )

    def forward(self, x):
        return self.model(x)
