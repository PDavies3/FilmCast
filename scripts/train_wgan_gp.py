"""
scripts/train_wgan_gp.py
---------------------------
WGAN-GP training for this repo's FiLM-conditioned, resolution-adaptive
generators -- adapted from "The GAN Book" (Kartik Chaudhary), Skill 7
(WGAN-GP) combined with Skill 8's conditional PatchGAN discriminator
design (models/critic_patchgan.py).

This is a SEPARATE, isolated training script -- it does not modify or
replace scripts/train.py. Use this one if the BCE-based GAN in train.py
is still unstable after the warmup/rain-weighting/GroupNorm changes;
WGAN-GP addresses discriminator collapse at its root (via the gradient
penalty enforcing a Lipschitz constraint) rather than through the
workarounds (label smoothing, lower D learning rate, spectral norm)
train.py relies on.

WHAT'S DIFFERENT FROM train.py:
  - Critic (PatchGANCritic), not the sigmoid discriminator -- outputs an
    unbounded per-patch realness score, trained on the WASSERSTEIN
    distance: critic_loss = mean(critic(fake)) - mean(critic(real)).
    Generator adversarial loss = -mean(critic(fake)) (make the critic
    score fakes as high/real as possible).
  - GRADIENT PENALTY (lambda_gp=10, matching the original WGAN-GP paper
    and the book's experiment): critic is scored on a random linear
    interpolation between real and fake fields; the penalty pushes the
    gradient of that score w.r.t. the interpolated field toward norm 1,
    which is what enforces the 1-Lipschitz constraint the Wasserstein
    distance requires -- WITHOUT weight clipping (WGAN-GP's whole
    contribution over plain WGAN) and without needing spectral norm.
  - N_CRITIC (default 5, matching the book/paper): the critic is updated
    5 times for every 1 generator update -- the critic needs to stay
    close to optimal for the Wasserstein distance estimate (and thus the
    generator's gradient from it) to be meaningful.
  - Optimizer: Adam(lr=1e-4, betas=(0, 0.9)) rather than the book's
    RMSprop(lr=0.00005) -- this is what the ORIGINAL WGAN-GP paper
    recommends over RMSprop specifically because momentum interacts
    poorly with the gradient penalty term; the book's simpler RMSprop
    choice is adequate for its unconditional MNIST experiment but Adam
    with these betas is the more broadly reliable choice here.

WHAT'S UNCHANGED FROM train.py: the generator itself (FiLMUNetGenerator /
FiLMAdaptiveGenerator, resolution-adaptive, FiLM-conditioned -- same
--architecture choices), the rain-weighted pixel loss and spatial-
gradient loss on the generator side (the book's WGAN-GP experiment is
PURE adversarial loss because it's unconditional generation; this repo's
task is conditional image-to-image, so the L1-style terms are combined
with the Wasserstein adversarial term -- this is standard pix2pix-style
practice, not a deviation from the WGAN-GP method itself), the
ConfigurableDownscalingDataset / config-driven pipeline, the validation
loop and --val_config / _best.pt checkpointing pattern, warmup epochs.

Real run:
    python scripts/train_wgan_gp.py --architecture unet \
        --config configs/quickstart_template.yaml \
        --val_config configs/quickstart_val.yaml \
        --epochs 50 --batch_size 4 --checkpoint_dir ./checkpoints_wgan_gp

Dry run (synthetic data, no real files needed):
    python scripts/train_wgan_gp.py --architecture unet --dry_run --epochs 1
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import ARCHITECTURES
from models.critic_patchgan import PatchGANCritic
from utils.conditioning import build_cond_vector
from utils.normalisation import normalize


# --- Shared loss helpers (duplicated from scripts/train.py rather than
# imported, so this script stays a fully isolated file -- see this repo's
# convention of preferring isolated new files over cross-script imports) ---

def gradient_loss(pred, target):
    """L1 loss on spatial finite-difference gradients -- penalizes
    speckle/discontinuity that per-pixel L1 misses."""
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    return F.l1_loss(pred_dy, target_dy) + F.l1_loss(pred_dx, target_dx)


def weighted_pixel_loss(pred, target, zero_precip_norm, rain_weight):
    """L1, upweighted where the real target has actual rain."""
    if zero_precip_norm is None or rain_weight <= 0:
        return F.l1_loss(pred, target)
    rain_mask = (target > zero_precip_norm).float()
    weights = 1.0 + rain_weight * rain_mask
    return torch.mean(weights * torch.abs(pred - target))


def compute_zero_precip_norm(target_norm_config):
    return float(normalize(np.array([0.0]), target_norm_config)[0])


def gradient_penalty(critic, real_field, fake_field, dynamic_in, static_in, device):
    """WGAN-GP's core contribution: score the critic on a random linear
    interpolation between real and fake fields, and penalize the
    gradient of that score (w.r.t. the interpolated FIELD only -- the
    conditioning context is held fixed) for deviating from norm 1. This
    enforces the 1-Lipschitz constraint the Wasserstein distance needs,
    without weight clipping."""
    batch_size = real_field.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real_field + (1 - alpha) * fake_field).requires_grad_(True)

    critic_scores = critic(interpolated, dynamic_in, static_in)

    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return torch.mean((gradient_norm - 1.0) ** 2)


def get_dry_run_batch(batch_size, dynamic_channels, static_channels, coarse_shape, fine_shape, device):
    dynamics = torch.rand(batch_size, dynamic_channels, *coarse_shape, device=device)
    statics = torch.rand(batch_size, static_channels, *fine_shape, device=device)
    real_rain = torch.rand(batch_size, 1, *fine_shape, device=device)
    valid_times = ["2020-06-15"] * batch_size
    lead_days = [5.0] * batch_size
    cond = build_cond_vector(valid_times, lead_days, max_lead_days=46.0).to(device)
    return dynamics, statics, real_rain, cond


def get_config_dataloader(config_path, batch_size, region_override=None, shuffle=True):
    from utils.config_loader import load_config
    from utils.folder_dataset import ConfigurableDownscalingDataset
    from torch.utils.data import DataLoader
    config = load_config(config_path)
    if region_override is not None:
        print(f">> --region override: using '{region_override}' instead of the config's region")
        config["region"] = region_override
    dataset = ConfigurableDownscalingDataset(config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=min(2, os.cpu_count() or 1))
    return config, dataset, loader


def _critic_step(critic, netG, dynamics, statics, real_rain, cond, optimizer_C, lambda_gp, device):
    with torch.no_grad():
        fake_rain = netG(dynamics, statics, cond)

    optimizer_C.zero_grad()
    critic_real = critic(real_rain, dynamics, statics).mean()
    critic_fake = critic(fake_rain, dynamics, statics).mean()
    gp = gradient_penalty(critic, real_rain, fake_rain, dynamics, statics, device)

    # Wasserstein critic loss: critic should score real HIGH, fake LOW --
    # minimizing (fake - real) pushes exactly that direction.
    loss_c = critic_fake - critic_real + lambda_gp * gp
    loss_c.backward()
    optimizer_C.step()

    return loss_c.item(), (critic_real - critic_fake).item(), gp.item()


def _generator_step(critic, netG, dynamics, statics, real_rain, cond, optimizer_G,
                     zero_precip_norm, rain_weight, pixel_weight, gradient_weight, use_adversarial):
    optimizer_G.zero_grad()
    fake_rain = netG(dynamics, statics, cond)

    loss_g_pixel = weighted_pixel_loss(fake_rain, real_rain, zero_precip_norm, rain_weight)
    loss_g_grad = gradient_loss(fake_rain, real_rain)

    if use_adversarial:
        # Generator wants the critic to score fakes HIGH -- minimizing
        # -critic(fake) does that.
        loss_g_adv = -critic(fake_rain, dynamics, statics).mean()
    else:
        loss_g_adv = torch.tensor(0.0)

    loss_g = loss_g_adv + pixel_weight * loss_g_pixel + gradient_weight * loss_g_grad
    loss_g.backward()
    optimizer_G.step()

    return loss_g.item(), loss_g_adv.item(), loss_g_pixel.item(), loss_g_grad.item()


@torch.no_grad()
def validate(netG, val_loader, device, zero_precip_norm, rain_weight):
    netG.eval()
    total_pixel, total_grad, n_batches = 0.0, 0.0, 0
    for batch in val_loader:
        dynamics = batch["dynamic_input"].to(device)
        statics = batch["static_input"].to(device)
        real_rain = batch["target"].to(device)
        cond = batch["cond"].to(device)
        fake_rain = netG(dynamics, statics, cond)
        total_pixel += weighted_pixel_loss(fake_rain, real_rain, zero_precip_norm, rain_weight).item()
        total_grad += gradient_loss(fake_rain, real_rain).item()
        n_batches += 1
    netG.train()
    return total_pixel / max(n_batches, 1), total_grad / max(n_batches, 1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Training architecture='{args.architecture}' (WGAN-GP) on {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.dry_run:
        print(">> DRY RUN: using synthetic data, no real files needed.")
        dynamic_channels, static_channels = 6, 2
        coarse_shape, fine_shape = (51, 55), (128, 128)

        ModelCls = ARCHITECTURES[args.architecture]
        netG = ModelCls(dynamic_in_channels=dynamic_channels, static_in_channels=static_channels, cond_dim=3).to(device)
        critic = PatchGANCritic(field_channels=1, dynamic_channels=dynamic_channels,
                                 static_channels=static_channels).to(device)

        optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.0, 0.9))
        optimizer_C = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))

        for epoch in range(args.epochs):
            use_adversarial = epoch >= args.warmup_epochs
            for step in range(args.dry_run_steps):
                dynamics, statics, real_rain, cond = get_dry_run_batch(
                    args.batch_size, dynamic_channels, static_channels, coarse_shape, fine_shape, device
                )
                if use_adversarial:
                    for _ in range(args.n_critic):
                        loss_c, wasserstein_dist, gp = _critic_step(
                            critic, netG, dynamics, statics, real_rain, cond, optimizer_C, args.lambda_gp, device
                        )
                else:
                    loss_c, wasserstein_dist, gp = 0.0, 0.0, 0.0
                loss_g, loss_g_adv, loss_g_pixel, loss_g_grad = _generator_step(
                    critic, netG, dynamics, statics, real_rain, cond, optimizer_G,
                    None, args.rain_weight, args.pixel_weight, args.gradient_weight, use_adversarial
                )
            tag = "(warmup, no adversarial) " if not use_adversarial else ""
            print(f"Epoch [{epoch+1}/{args.epochs}] (dry run) {tag}"
                  f"Loss_C: {loss_c:.4f} | Wasserstein_D: {wasserstein_dist:.4f} | GP: {gp:.4f} | "
                  f"Loss_G_pixel: {loss_g_pixel:.4f} | Loss_G_grad: {loss_g_grad:.4f}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.architecture}_wgan_gp_dryrun.pt")
        torch.save({"generator_state_dict": netG.state_dict(), "architecture": args.architecture}, ckpt_path)
        print(f">> Dry run complete. Checkpoint saved to {ckpt_path}")
        return

    print(f">> Using config-driven dataset: {args.config}")
    config, dataset, loader = get_config_dataloader(args.config, args.batch_size, region_override=args.region)
    zero_precip_norm = compute_zero_precip_norm(config["target"]["normalisation"])
    print(f">> Zero-precip normalized value: {zero_precip_norm:.4f} "
          f"(rain_weight={args.rain_weight} applied above this threshold)")

    val_loader = None
    if args.val_config:
        print(f">> Using validation config: {args.val_config}")
        _, val_dataset, val_loader = get_config_dataloader(args.val_config, args.batch_size, shuffle=False)
        print(f">> {len(val_dataset)} validation samples")

    probe_batch = next(iter(loader))
    dynamic_channels = probe_batch["dynamic_input"].shape[1]
    static_channels = probe_batch["static_input"].shape[1]

    ModelCls = ARCHITECTURES[args.architecture]
    netG = ModelCls(dynamic_in_channels=dynamic_channels, static_in_channels=static_channels, cond_dim=3).to(device)
    critic = PatchGANCritic(field_channels=1, dynamic_channels=dynamic_channels,
                             static_channels=static_channels).to(device)

    if args.resume_from:
        print(f">> Resuming generator weights from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        netG.load_state_dict(ckpt["generator_state_dict"])
        print(f"   Resumed successfully from epoch {ckpt.get('epoch', '?')} of the source run.")

    # P.Davies: add -- betas=(0, 0.9), not this repo's usual (0.5, 0.999):
    # the WGAN-GP paper specifically found momentum (high beta1) destabilizes
    # training when combined with the gradient penalty term.
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.0, 0.9))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))

    best_val_pixel_loss = float("inf")

    for epoch in range(args.epochs):
        use_adversarial = epoch >= args.warmup_epochs
        netG.train()
        critic.train()

        for i, batch in enumerate(loader):
            dynamics = batch["dynamic_input"].to(device)
            statics = batch["static_input"].to(device)
            real_rain = batch["target"].to(device)
            cond = batch["cond"].to(device)

            if use_adversarial:
                for _ in range(args.n_critic):
                    loss_c, wasserstein_dist, gp = _critic_step(
                        critic, netG, dynamics, statics, real_rain, cond, optimizer_C, args.lambda_gp, device
                    )
            else:
                loss_c, wasserstein_dist, gp = 0.0, 0.0, 0.0

            loss_g, loss_g_adv, loss_g_pixel, loss_g_grad = _generator_step(
                critic, netG, dynamics, statics, real_rain, cond, optimizer_G,
                zero_precip_norm, args.rain_weight, args.pixel_weight, args.gradient_weight, use_adversarial
            )

        tag = "(warmup, no adversarial) " if not use_adversarial else ""
        print(f"Epoch [{epoch+1}/{args.epochs}] {tag}"
              f"Loss_C: {loss_c:.4f} | Wasserstein_D: {wasserstein_dist:.4f} | GP: {gp:.4f} | "
              f"Loss_G_adv: {loss_g_adv:.4f} | Loss_G_pixel: {loss_g_pixel:.4f} | Loss_G_grad: {loss_g_grad:.4f}")

        if val_loader is not None:
            val_pixel_loss, val_grad_loss = validate(netG, val_loader, device, zero_precip_norm, args.rain_weight)
            print(f"           Val_pixel: {val_pixel_loss:.4f} | Val_grad: {val_grad_loss:.4f}")
            if val_pixel_loss < best_val_pixel_loss:
                best_val_pixel_loss = val_pixel_loss
                best_path = os.path.join(args.checkpoint_dir, f"{args.architecture}_wgan_gp_best.pt")
                torch.save({
                    "generator_state_dict": netG.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "architecture": args.architecture,
                    "epoch": epoch + 1,
                    "val_pixel_loss": val_pixel_loss,
                }, best_path)
                print(f"           >> New best (val_pixel={val_pixel_loss:.4f}). Saved to {best_path}")

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.architecture}_wgan_gp_epoch{epoch+1}.pt")
            torch.save({
                "generator_state_dict": netG.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "architecture": args.architecture,
                "epoch": epoch + 1,
            }, ckpt_path)
            print(f">> Checkpoint saved to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a FiLM-conditioned, resolution-adaptive downscaling generator "
                     "using WGAN-GP (Wasserstein loss + gradient penalty), adapted from "
                     "'The GAN Book' Skill 7."
    )
    parser.add_argument("--architecture", type=str, default="unet", choices=list(ARCHITECTURES.keys()))
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--val_config", type=str, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5,
                         help="Epochs of pixel+gradient loss only before the critic/adversarial "
                              "term is introduced. Default: 5.")
    parser.add_argument("--n_critic", type=int, default=5,
                         help="Critic updates per generator update. Default: 5, matching the "
                              "WGAN-GP paper and book -- the critic needs to stay near-optimal "
                              "for its Wasserstein estimate to give the generator a useful gradient.")
    parser.add_argument("--lambda_gp", type=float, default=10.0,
                         help="Gradient penalty weight. Default: 10, matching the WGAN-GP paper.")
    parser.add_argument("--pixel_weight", type=float, default=100.0)
    parser.add_argument("--gradient_weight", type=float, default=50.0)
    parser.add_argument("--rain_weight", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="Default: 1e-4 (WGAN-GP paper's recommendation, used for BOTH "
                              "generator and critic -- unlike train.py's asymmetric LR workaround, "
                              "the gradient penalty makes that asymmetry unnecessary here).")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_wgan_gp")
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_steps", type=int, default=3)
    args = parser.parse_args()

    if not args.dry_run and args.config is None:
        parser.error("--config is required unless --dry_run is set")

    train(args)


if __name__ == "__main__":
    main()
