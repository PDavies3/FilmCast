"""
scripts/train.py
------------------
Training script for the FiLM-conditioned, resolution-adaptive downscaling
GANs in this repo.

Real run:
    python scripts/train.py --architecture unet --data_dir <path> \
        --batch_size 8 --epochs 20 --checkpoint_dir ./checkpoints

Dry run (synthetic data, no real files needed -- sanity-checks the training
loop mechanics: optimizer steps, loss computation, checkpoint saving):
    python scripts/train.py --architecture unet --dry_run --epochs 1
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import get_model
from utils.conditioning import build_cond_vector


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


def get_real_dataloader(data_root, batch_size):
    from utils.grib_dataset import S2SGribDataset
    from torch.utils.data import DataLoader
    dataset = S2SGribDataset(data_root=data_root, split="train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=min(2, os.cpu_count() or 1))
    return dataset, loader


def batch_to_cond(batch, canonical_steps, max_lead_days, device):
    valid_times = batch["meta"]["valid_time"]
    step_indices = batch["meta"]["canonical_step_idx"]
    lead_days = [canonical_steps[i].astype("timedelta64[D]").astype(float) for i in step_indices]
    return build_cond_vector(valid_times, lead_days, max_lead_days).to(device)


def _train_step(netG, netD, dynamics, statics, real_rain, cond, optimizer_G, optimizer_D, criterion_gan, criterion_pixel):
    optimizer_D.zero_grad()
    fake_rain = netG(dynamics, statics, cond)
    logits_real = netD(real_rain)
    logits_fake = netD(fake_rain.detach())
    loss_d_real = criterion_gan(logits_real, torch.ones_like(logits_real))
    loss_d_fake = criterion_gan(logits_fake, torch.zeros_like(logits_fake))
    loss_d = (loss_d_real + loss_d_fake) * 0.5
    loss_d.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()
    logits_fake_for_g = netD(fake_rain)
    loss_g_adv = criterion_gan(logits_fake_for_g, torch.ones_like(logits_fake_for_g))
    loss_g_pixel = criterion_pixel(fake_rain, real_rain)
    loss_g = loss_g_adv + 10.0 * loss_g_pixel
    loss_g.backward()
    optimizer_G.step()

    return loss_d.item(), loss_g.item()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Training architecture='{args.architecture}' on {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()

    if args.dry_run:
        print(">> DRY RUN: using synthetic data, no real files needed.")
        dynamic_channels, static_channels = 6, 2
        coarse_shape, fine_shape = (51, 55), (128, 128)
        netG, netD = get_model(args.architecture, dynamic_channels, static_channels, device)
        optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

        for epoch in range(args.epochs):
            for step in range(args.dry_run_steps):
                dynamics, statics, real_rain, cond = get_dry_run_batch(
                    args.batch_size, dynamic_channels, static_channels, coarse_shape, fine_shape, device
                )
                loss_d, loss_g = _train_step(netG, netD, dynamics, statics, real_rain, cond,
                                              optimizer_G, optimizer_D, criterion_gan, criterion_pixel)
            print(f"Epoch [{epoch+1}/{args.epochs}] (dry run) Loss_D: {loss_d:.4f} | Loss_G: {loss_g:.4f}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.architecture}_dryrun.pt")
        torch.save({"generator_state_dict": netG.state_dict(), "architecture": args.architecture}, ckpt_path)
        print(f">> Dry run complete. Checkpoint saved to {ckpt_path}")
        return

    dataset, loader = get_real_dataloader(args.data_dir, args.batch_size)
    probe_batch = next(iter(loader))
    dynamic_channels = probe_batch["dynamic_input"].shape[1]
    static_channels = probe_batch["static_input"].shape[1]
    max_lead_days = float(dataset.canonical_steps.max().astype("timedelta64[D]").astype(float))

    netG, netD = get_model(args.architecture, dynamic_channels, static_channels, device)
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        netG.train()
        netD.train()
        for i, batch in enumerate(loader):
            dynamics = batch["dynamic_input"].to(device)
            statics = batch["static_input"].to(device)
            real_rain = batch["target_imerg"].to(device)
            cond = batch_to_cond(batch, dataset.canonical_steps, max_lead_days, device)

            loss_d, loss_g = _train_step(netG, netD, dynamics, statics, real_rain, cond,
                                          optimizer_G, optimizer_D, criterion_gan, criterion_pixel)

        print(f"Epoch [{epoch+1}/{args.epochs}] Loss_D: {loss_d:.4f} | Loss_G: {loss_g:.4f}")

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
    parser.add_argument("--architecture", type=str, required=True, choices=["residual", "unet"])
    parser.add_argument("--data_dir", type=str, default=None, help="Required unless --dry_run")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_steps", type=int, default=3)
    args = parser.parse_args()

    if not args.dry_run and args.data_dir is None:
        parser.error("--data_dir is required unless --dry_run is set")

    train(args)


if __name__ == "__main__":
    main()
