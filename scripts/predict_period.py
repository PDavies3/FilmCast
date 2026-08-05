"""
scripts/predict_period.py
---------------------------
Runs inference over an ENTIRE config-driven dataset (e.g. every date in a
held-out period defined by start_date/end_date), rather than one
--sample_index at a time like scripts/infer.py. One NetCDF prediction file
per sample, batched through the model for speed.

Predictions are saved in PHYSICAL units (e.g. mm/day), not the raw
normalized model output -- denormalize() is applied using the same
target normalisation config the model was trained against, followed by
a non-negativity clip (physical precipitation can't be negative; the
network's raw output CAN be, since it operates in normalized space where
zero precipitation is not zero -- see models/film_unet_gan.py's removed
final ReLU for why that clip no longer lives inside the network).

Predictions also carry the dataset's real lat/lon coordinates (from
ConfigurableDownscalingDataset.__getitem__), so output NetCDFs are
properly georeferenced (latitude/longitude dims) instead of anonymous
y/x indices.

Real run:
    python scripts/predict_period.py --architecture unet \
        --checkpoint ./checkpoints/unet_best.pt \
        --config configs/quickstart_inference.yaml \
        --output_dir ./predictions --batch_size 8

Deliberately no --dry_run here -- scripts/infer.py's --dry_run already
covers "does the model run at all" with synthetic data. This script's
whole job is real, config-driven, multi-sample inference over a period.
"""
import argparse
import os
import sys

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import ARCHITECTURES
from utils.normalisation import denormalize


def load_generator(checkpoint_path, architecture, dynamic_channels, static_channels, device):
    ModelCls = ARCHITECTURES[architecture]
    netG = ModelCls(dynamic_in_channels=dynamic_channels, static_in_channels=static_channels, cond_dim=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    netG.load_state_dict(ckpt["generator_state_dict"])
    netG.eval()
    return netG


def save_prediction(prediction, lat, lon, output_path):
    ds = xr.Dataset(
        {"precipitation": (("latitude", "longitude"), prediction)},
        coords={"latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(output_path)


def run(args):
    from utils.config_loader import load_config
    from utils.folder_dataset import ConfigurableDownscalingDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Running batched inference on {device}")

    config = load_config(args.config)
    if args.region is not None:
        print(f">> --region override: using '{args.region}' instead of the config's region")
        config["region"] = args.region

    dataset = ConfigurableDownscalingDataset(config)
    print(f">> {len(dataset)} samples in this dataset (period: "
          f"{config.get('start_date', 'earliest')} to {config.get('end_date', 'latest')})")
    target_norm_config = config["target"]["normalisation"]

    probe = dataset[0]
    dynamic_channels = probe["dynamic_input"].shape[0]
    static_channels = probe["static_input"].shape[0]

    netG = load_generator(args.checkpoint, args.architecture, dynamic_channels, static_channels, device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=min(2, os.cpu_count() or 1))

    os.makedirs(args.output_dir, exist_ok=True)

    n_saved = 0
    with torch.no_grad():
        for batch in loader:
            dynamics = batch["dynamic_input"].to(device)
            statics = batch["static_input"].to(device)
            cond = batch["cond"].to(device)

            predictions = netG(dynamics, statics, cond)  # [B, 1, H, W] -- normalized

            batch_size = predictions.shape[0]
            dates = batch["meta"]["date"]
            lead_hours = batch["meta"]["lead_hours"]
            members = batch["meta"]["member"]

            for i in range(batch_size):
                out_name = f"pred_{dates[i]}_lead{int(lead_hours[i])}_m{int(members[i])}.nc"
                out_path = os.path.join(args.output_dir, out_name)

                # P.Davies: add -- convert normalized model output to
                # physical units, then clip to physically valid range.
                pred_physical = denormalize(predictions[i, 0].cpu().numpy(), target_norm_config)
                pred_physical = np.clip(pred_physical, a_min=0, a_max=None)

                save_prediction(
                    pred_physical,
                    batch["lat"][i].numpy(),
                    batch["lon"][i].numpy(),
                    out_path,
                )
                n_saved += 1

            print(f">> Saved {n_saved}/{len(dataset)} predictions...", end="\r")

    print(f"\n>> Done. {n_saved} predictions saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference over every sample in a config-driven dataset "
                     "(e.g. a full held-out period), saving one georeferenced, "
                     "physical-units NetCDF per sample."
    )
    parser.add_argument("--architecture", type=str, default="unet", choices=["residual", "unet"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--region", type=str, default=None,
                         help="Override the config's region without editing the YAML.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./predictions")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
