"""
models/__init__.py
--------------------
Model factory. Two FiLM-conditioned, resolution-adaptive architectures:

    "residual" -> FiLMAdaptiveGenerator (models/film_adaptive_gan.py)
    "unet"     -> FiLMUNetGenerator      (models/film_unet_gan.py)

Both take forward(dynamic_in, static_in, cond) -- cond is the
[lead_time_norm, sin_doy, cos_doy] vector from utils/conditioning.py.
"""
from models.film_adaptive_gan import FiLMAdaptiveGenerator
from models.film_unet_gan import FiLMUNetGenerator
from models.discriminator import Pix2PixDiscriminator

ARCHITECTURES = {
    "residual": FiLMAdaptiveGenerator,
    "unet": FiLMUNetGenerator,
}


def get_model(architecture, dynamic_channels, static_channels, device, cond_dim=3):
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Choose from: {list(ARCHITECTURES.keys())}"
        )
    ModelCls = ARCHITECTURES[architecture]
    netG = ModelCls(
        dynamic_in_channels=dynamic_channels,
        static_in_channels=static_channels,
        cond_dim=cond_dim,
    ).to(device)
    netD = Pix2PixDiscriminator(in_channels=1).to(device)
    return netG, netD
