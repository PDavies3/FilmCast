"""
scripts/train.py
------------------
Training script for the FiLM-conditioned, resolution-adaptive downscaling
GANs in this repo.

LOSS DESIGN (generator):
    loss_g = loss_g_adv + pixel_weight * loss_g_pixel + gradient_weight * loss_g_grad

  - loss_g_pixel is a WEIGHTED L1: pixels where the real target has actual
    rain (normalized value above the exact normalized-zero-precip value,
    computed from the target's own normalisation config) are weighted
    rain_weight times more than dry pixels. Precipitation fields are
    extremely zero-inflated -- most pixels are dry -- so plain L1 lets a
    model get a deceptively low loss by hedging toward near-zero
    everywhere rather than actually resolving real rain events.
  - loss_g_grad is a spatial-gradient (finite-difference) L1 loss --
    penalizes the predicted field disagreeing sharply with its own
    neighbors, which per-pixel L1 doesn't capture at all.

WARM-UP (--warmup_epochs, default 5): the discriminator is not trained
and loss_g_adv is excluded for the first N epochs, so the generator
learns real spatial/intensity structure from pixel+gradient loss alone
before an initially-untrained discriminator can contribute a gradient.

VALIDATION (--val_config, optional): if given, a second config-driven
dataset (typically a held-out DATE RANGE within/adjacent to the training
period, NOT the final test/inference period) is evaluated (no gradient)
at the end of every epoch. The checkpoint with the lowest validation
pixel loss is kept separately as {architecture}_best.pt, alongside the
regular periodic checkpoints -- so "best" is chosen by held-out
performance, not just whichever epoch happened to run last.

Real run (config-driven, recommended):
    python scripts/train.py --architecture unet --config configs/ghana_precip.yml \
        --val_config configs/ghana_precip_val.yml \
        --batch_size 4 --epochs 50 --warmup_epochs 5 --checkpoint_dir ./checkpoints

Real run (legacy GRIB path):
    python scripts/train.py --architecture unet --data_dir <path> \
        --batch_size 8 --epochs 20 --checkpoint_dir ./checkpoints

Dry run (synthetic data, no real files needed -- sanity-checks the training
loop mechanics: optimizer steps, loss computation, checkpoint saving):
    python scripts/train.py --architecture unet --dry_run --epochs 1
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import get_model
from utils.conditioning import build_cond_vector
from utils.normalisation import normalize


def gradient_loss(pred, target):
    """L1 loss on spatial finite-difference gradients between predicted
    and target fields -- penalizes speckle/discontinuity that plain
    per-pixel L1 misses entirely."""
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    return F.l1_loss(pred_dy, target_dy) + F.l1_loss(pred_dx, target_dx)


def weighted_pixel_loss(pred, target, zero_precip_norm, rain_weight):
    """L1, upweighted at pixels where the real target has actual rain.
    If zero_precip_norm is None (e.g. legacy GRIB path with no config
    normalisation available), falls back to plain L1."""
    if zero_precip_norm is None or rain_weight <= 0:
        return F.l1_loss(pred, target)
    rain_mask = (target > zero_precip_norm).float()
    weights = 1.0 + rain_weight * rain_mask
    return torch.mean(weights * torch.abs(pred - target))


def compute_zero_precip_norm(target_norm_config):
    """The exact normalized value corresponding to physically zero
    precipitation, computed generically via utils.normalisation.normalize
    (works for either zscore or log1p -- no per-type special-casing
    needed here)."""
    return float(normalize(np.array([0.0]), target_norm_config)[0])


def get_dry_run_batch(batch_size, dynamic_channels, static_channels, coarse_shape, fine_shape, device):
    """Synthetic batch of the right shapes -- verifies training loop
    mechanics without needing real GRIB/IMERG files."""
    dynamics = torch.rand(batch_size, dynamic_channels, *coarse_shape, device=device)
    statics = torch.rand(batch_size, static_channels, *fine_shape, device=device)
    real_rain = torch.rand(batch_size, 1, *fine_shape, device=device)
    valid_times = ["2020-06-15"] * batch_size
    lead_days = [5.0] * batch_size
    cond = build_cond_vector(valid_times, lead_days, max_lead_days=46.0).to(device)
    return dynamics, statics, real_rain, cond


def get_grib_dataloader(data_root, batch_size):
    """Legacy path: single consolidated GRIB file."""
    from utils.grib_dataset import S2SGribDataset
    from torch.utils.data import DataLoader
    dataset = S2SGribDataset(data_root=data_root, split="train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=min(2, os.cpu_count() or 1))
    return dataset, loader


def get_config_dataloader(config_path, batch_size, region_override=None, shuffle=True):
    """Config-driven path: per-variable-folder, per-day-file layout."""
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


def grib_batch_to_cond(batch, canonical_steps, max_lead_days, device):
    valid_times = batch["meta"]["valid_time"]
    step_indices = batch["meta"]["canonical_step_idx"]
    lead_days = [canonical_steps[i].astype("timedelta64[D]").astype(float) for i in step_indices]
    return build_cond_vector(valid_times, lead_days, max_lead_days).to(device)


def _train_step(netG, netD, dynamics, statics, real_rain, cond, optimizer_G, optimizer_D,
                 criterion_gan, zero_precip_norm, rain_weight, pixel_weight, gradient_weight,
                 use_adversarial):
    fake_rain = netG(dynamics, statics, cond)

    if use_adversarial:
        optimizer_D.zero_grad()
        logits_real = netD(real_rain)
        logits_fake = netD(fake_rain.detach())
        # One-sided label smoothing on the REAL label only -- standard GAN
        # stabilization trick, makes it harder for D to become
        # overconfident and collapse toward a trivial solution.
        loss_d_real = criterion_gan(logits_real, torch.full_like(logits_real, 0.9))
        loss_d_fake = criterion_gan(logits_fake, torch.zeros_like(logits_fake))
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        optimizer_D.step()
    else:
        loss_d = torch.tensor(0.0)

    optimizer_G.zero_grad()
    loss_g_pixel = weighted_pixel_loss(fake_rain, real_rain, zero_precip_norm, rain_weight)
    loss_g_grad = gradient_loss(fake_rain, real_rain)

    if use_adversarial:
        logits_fake_for_g = netD(fake_rain)
        loss_g_adv = criterion_gan(logits_fake_for_g, torch.ones_like(logits_fake_for_g))
    else:
        loss_g_adv = torch.tensor(0.0)

    loss_g = loss_g_adv + pixel_weight * loss_g_pixel + gradient_weight * loss_g_grad
    loss_g.backward()
    optimizer_G.step()

    return loss_d.item(), loss_g.item(), loss_g_adv.item(), loss_g_pixel.item(), loss_g_grad.item()


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
    print(f">> Training architecture='{args.architecture}' on {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    criterion_gan = nn.BCEWithLogitsLoss()

    if args.dry_run:
        print(">> DRY RUN: using synthetic data, no real files needed.")
        dynamic_channels, static_channels = 6, 2
        coarse_shape, fine_shape = (51, 55), (128, 128)
        netG, netD = get_model(args.architecture, dynamic_channels, static_channels, device)
        optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))

        for epoch in range(args.epochs):
            use_adversarial = epoch >= args.warmup_epochs
            for step in range(args.dry_run_steps):
                dynamics, statics, real_rain, cond = get_dry_run_batch(
                    args.batch_size, dynamic_channels, static_channels, coarse_shape, fine_shape, device
                )
                loss_d, loss_g, loss_g_adv, loss_g_pixel, loss_g_grad = _train_step(
                    netG, netD, dynamics, statics, real_rain, cond, optimizer_G, optimizer_D,
                    criterion_gan, None, args.rain_weight, args.pixel_weight, args.gradient_weight,
                    use_adversarial
                )
            tag = "(warmup, no adversarial) " if not use_adversarial else ""
            print(f"Epoch [{epoch+1}/{args.epochs}] (dry run) {tag}"
                  f"Loss_D: {loss_d:.4f} | Loss_G_adv: {loss_g_adv:.4f} | "
                  f"Loss_G_pixel: {loss_g_pixel:.4f} | Loss_G_grad: {loss_g_grad:.4f}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.architecture}_dryrun.pt")
        torch.save({"generator_state_dict": netG.state_dict(), "architecture": args.architecture}, ckpt_path)
        print(f">> Dry run complete. Checkpoint saved to {ckpt_path}")
        return

    use_config = args.config is not None
    zero_precip_norm = None
    val_loader = None

    if use_config:
        print(f">> Using config-driven dataset: {args.config}")
        config, dataset, loader = get_config_dataloader(args.config, args.batch_size, region_override=args.region)
        zero_precip_norm = compute_zero_precip_norm(config["target"]["normalisation"])
        print(f">> Zero-precip normalized value: {zero_precip_norm:.4f} "
              f"(rain_weight={args.rain_weight} applied above this threshold)")

        if args.val_config:
            print(f">> Using validation config: {args.val_config}")
            _, val_dataset, val_loader = get_config_dataloader(args.val_config, args.batch_size, shuffle=False)
            print(f">> {len(val_dataset)} validation samples")
    else:
        print(f">> Using legacy GRIB dataset: {args.data_dir}")
        dataset, loader = get_grib_dataloader(args.data_dir, args.batch_size)
        max_lead_days = float(dataset.canonical_steps.max().astype("timedelta64[D]").astype(float))

    probe_batch = next(iter(loader))
    dynamic_channels = probe_batch["dynamic_input"].shape[1]
    static_channels = probe_batch["static_input"].shape[1]

    netG, netD = get_model(args.architecture, dynamic_channels, static_channels, device)

    if args.resume_from:
        print(f">> Resuming weights from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        try:
            netG.load_state_dict(ckpt["generator_state_dict"])
        except RuntimeError as e:
            raise RuntimeError(
                f"Could not load generator weights from {args.resume_from} into this "
                f"model (dynamic_channels={dynamic_channels}, static_channels={static_channels}). "
                f"This usually means the input variable set differs between the checkpoint's "
                f"original config and the one you're resuming into -- weight transfer requires "
                f"the SAME number of input channels (e.g. same physical variables, even if "
                f"sourced from a different data pipeline like ERA5 vs IFS/S2S). "
                f"Original error: {e}"
            ) from e
        if "discriminator_state_dict" in ckpt:
            try:
                netD.load_state_dict(ckpt["discriminator_state_dict"])
            except RuntimeError as e:
                print(f"   WARNING: could not resume discriminator weights ({e}). "
                      f"Continuing with a freshly-initialized discriminator -- this is "
                      f"usually fine since the discriminator only ever sees the 1-channel "
                      f"precipitation field regardless of input variable set.")
        print(f"   Resumed successfully from epoch {ckpt.get('epoch', '?')} of the source run.")

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))

    best_val_pixel_loss = float("inf")

    for epoch in range(args.epochs):
        use_adversarial = epoch >= args.warmup_epochs
        netG.train()
        netD.train()
        for i, batch in enumerate(loader):
            dynamics = batch["dynamic_input"].to(device)
            statics = batch["static_input"].to(device)

            if use_config:
                real_rain = batch["target"].to(device)
                cond = batch["cond"].to(device)
            else:
                real_rain = batch["target_imerg"].to(device)
                cond = grib_batch_to_cond(batch, dataset.canonical_steps, max_lead_days, device)

            loss_d, loss_g, loss_g_adv, loss_g_pixel, loss_g_grad = _train_step(
                netG, netD, dynamics, statics, real_rain, cond, optimizer_G, optimizer_D,
                criterion_gan, zero_precip_norm, args.rain_weight, args.pixel_weight,
                args.gradient_weight, use_adversarial
            )

        tag = "(warmup, no adversarial) " if not use_adversarial else ""
        print(f"Epoch [{epoch+1}/{args.epochs}] {tag}"
              f"Loss_D: {loss_d:.4f} | Loss_G_adv: {loss_g_adv:.4f} | "
              f"Loss_G_pixel: {loss_g_pixel:.4f} | Loss_G_grad: {loss_g_grad:.4f}")

        if val_loader is not None:
            val_pixel_loss, val_grad_loss = validate(netG, val_loader, device, zero_precip_norm, args.rain_weight)
            print(f"           Val_pixel: {val_pixel_loss:.4f} | Val_grad: {val_grad_loss:.4f}")
            if val_pixel_loss < best_val_pixel_loss:
                best_val_pixel_loss = val_pixel_loss
                best_path = os.path.join(args.checkpoint_dir, f"{args.architecture}_best.pt")
                torch.save({
                    "generator_state_dict": netG.state_dict(),
                    "discriminator_state_dict": netD.state_dict(),
                    "architecture": args.architecture,
                    "epoch": epoch + 1,
                    "val_pixel_loss": val_pixel_loss,
                }, best_path)
                print(f"           >> New best (val_pixel={val_pixel_loss:.4f}). Saved to {best_path}")

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.architecture}_epoch{epoch+1}.pt")
            torch.save({
                "generator_state_dict": netG.state_dict(),
                "discriminator_state_dict": netD.state_dict(),
                "architecture": args.architecture,
                "epoch": epoch + 1,
            }, ckpt_path)
            print(f">> Checkpoint saved to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a FiLM-conditioned, resolution-adaptive downscaling GAN")
    parser.add_argument("--architecture", type=str, default="unet", choices=["residual", "unet"],
                         help="Default: unet")
    parser.add_argument("--config", type=str, default=None,
                         help="Path to a YAML config (recommended). See configs/ for examples.")
    parser.add_argument("--val_config", type=str, default=None,
                         help="Optional YAML config for a held-out VALIDATION period (e.g. the "
                              "last month or two of the training year). Evaluated each epoch; the "
                              "lowest-val-loss checkpoint is saved as {architecture}_best.pt. "
                              "Should be a different date range than --config, and different "
                              "from whatever you use as your final test/inference set.")
    parser.add_argument("--data_dir", type=str, default=None,
                         help="Legacy: path to a single consolidated GRIB file. Ignored if --config is given.")
    parser.add_argument("--region", type=str, default=None,
                         help="Override the config's region without editing the YAML, e.g. "
                              "--region ghana / --region west_africa / --region africa, or "
                              "any preset name from utils/regions.py. Only applies with --config.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=5,
                         help="Epochs of pixel+gradient loss only, discriminator not trained, "
                              "before adversarial training starts. Default: 5.")
    parser.add_argument("--pixel_weight", type=float, default=100.0,
                         help="Weight on L1 pixel loss in the generator's total loss. Default: 100 "
                              "(higher than standard pix2pix's 10 -- precipitation fields are sparse "
                              "and skewed, so a weak pixel term lets the adversarial term dominate).")
    parser.add_argument("--gradient_weight", type=float, default=50.0,
                         help="Weight on the spatial-gradient (finite-difference) loss -- penalizes "
                              "the predicted field disagreeing with its own neighbors, targeting "
                              "speckle/discontinuity directly. Default: 50.")
    parser.add_argument("--rain_weight", type=float, default=5.0,
                         help="Extra weight applied to pixel loss where the real target has "
                              "actual rain (vs. dry pixels). Default: 5. Set to 0 to disable "
                              "(plain unweighted L1).")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--resume_from", type=str, default=None,
                         help="Path to a checkpoint to resume generator (and discriminator, if "
                              "compatible) weights from. Used for e.g. ERA5-pretrain -> IFS/S2S "
                              "fine-tune transitions -- requires the SAME input channel count "
                              "between the checkpoint's original config and this run's.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_steps", type=int, default=3)
    args = parser.parse_args()

    if not args.dry_run and args.config is None and args.data_dir is None:
        parser.error("--config or --data_dir is required unless --dry_run is set")

    train(args)


if __name__ == "__main__":
    main()
