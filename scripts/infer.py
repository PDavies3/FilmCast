"""
scripts/infer.py
------------------
Loads a checkpoint from scripts/train.py and runs inference, saving the
downscaled precipitation output as a NetCDF file. Two data-source modes,
matching scripts/train.py:

  --config <path/to/config.yml>   RECOMMENDED. Config-driven, reads the
      per-variable-folder, per-day-file layout.
  --data_dir <path>                Legacy: single consolidated GRIB file.

Real run (config-driven, recommended):
    python scripts/infer.py --architecture unet --checkpoint ./checkpoints/unet_epoch20.pt \
        --config configs/ghana_precip.yml --sample_index 0 --output_dir ./predictions

Real run (legacy GRIB path):
    python scripts/infer.py --architecture unet --checkpoint ./checkpoints/unet_epoch20.pt \
        --data_dir <path> --sample_index 0 --output_dir ./predictions

Dry run (synthetic input, no real files or checkpoint needed):
    python scripts/infer.py --architecture unet --dry_run --output_dir ./predictions
"""
import argparse
import os
import sys

import torch
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import ARCHITECTURES
from utils.conditioning import build_cond_vector


def load_generator(checkpoint_path, architecture, dynamic_channels, static_channels, device):
    ModelCls = ARCHITECTURES[architecture]
    netG = ModelCls(dynamic_in_channels=dynamic_channels, static_in_channels=static_channels, cond_dim=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    netG.load_state_dict(ckpt["generator_state_dict"])
    netG.eval()
    return netG


def save_prediction(prediction, output_path, lat=None, lon=None):
    if lat is not None and lon is not None:
        coords = {"latitude": lat, "longitude": lon}
        dims = ("latitude", "longitude")
    else:
        coords = {}
        dims = ("y", "x")
    ds = xr.Dataset({"precipitation": (dims, prediction)}, coords=coords)
    ds.to_netcdf(output_path)


def run_dry_run(args, device):
    print(">> DRY RUN: using an untrained model and synthetic input, no real files needed.")
    dynamic_channels, static_channels = 6, 2
    coarse_shape, fine_shape = (51, 55), (128, 128)

    ModelCls = ARCHITECTURES[args.architecture]
    netG = ModelCls(dynamic_in_channels=dynamic_channels, static_in_channels=static_channels, cond_dim=3).to(device)
    netG.eval()

    dynamics = torch.rand(1, dynamic_channels, *coarse_shape, device=device)
    statics = torch.rand(1, static_channels, *fine_shape, device=device)
    cond = build_cond_vector(["2020-06-15"], [5.0], max_lead_days=46.0).to(device)

    with torch.no_grad():
        prediction = netG(dynamics, statics, cond)

    print("Prediction shape:", prediction.shape)
    assert torch.all(prediction >= 0), "Non-negativity constraint violated"

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.architecture}_dryrun_prediction.nc")
    save_prediction(prediction[0, 0].cpu().numpy(), out_path)
    print(f">> Dry run complete. Prediction saved to {out_path}")


def run_config_inference(args, device):
    from utils.config_loader import load_config
    from utils.folder_dataset import ConfigurableDownscalingDataset

    config = load_config(args.config)
    if args.region is not None:
        print(f">> --region override: using '{args.region}' instead of the config's region")
        config["region"] = args.region
    dataset = ConfigurableDownscalingDataset(config)
    sample = dataset[args.sample_index]

    dynamic_channels = sample["dynamic_input"].shape[0]
    static_channels = sample["static_input"].shape[0]
    netG = load_generator(args.checkpoint, args.architecture, dynamic_channels, static_channels, device)

    dynamics = sample["dynamic_input"].unsqueeze(0).to(device)
    statics = sample["static_input"].unsqueeze(0).to(device)
    cond = sample["cond"].unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = netG(dynamics, statics, cond)

    os.makedirs(args.output_dir, exist_ok=True)
    meta = sample["meta"]
    out_name = f"pred_{meta['date']}_lead{meta['lead_hours']}_m{meta['member']}.nc"
    out_path = os.path.join(args.output_dir, out_name)
    save_prediction(prediction[0, 0].cpu().numpy(), out_path)
    print(f">> Prediction saved to {out_path}")
    print(f">> Sample metadata: {meta}")


def run_grib_inference(args, device):
    from utils.grib_dataset import S2SGribDataset

    dataset = S2SGribDataset(data_root=args.data_dir, split=args.split)
    sample = dataset[args.sample_index]

    dynamic_channels = sample["dynamic_input"].shape[0]
    static_channels = sample["static_input"].shape[0]
    netG = load_generator(args.checkpoint, args.architecture, dynamic_channels, static_channels, device)

    dynamics = sample["dynamic_input"].unsqueeze(0).to(device)
    statics = sample["static_input"].unsqueeze(0).to(device)

    max_lead_days = float(dataset.canonical_steps.max().astype("timedelta64[D]").astype(float))
    step_idx = sample["meta"]["canonical_step_idx"]
    lead_days = dataset.canonical_steps[step_idx].astype("timedelta64[D]").astype(float)
    cond = build_cond_vector([sample["meta"]["valid_time"]], [lead_days], max_lead_days).to(device)

    with torch.no_grad():
        prediction = netG(dynamics, statics, cond)

    os.makedirs(args.output_dir, exist_ok=True)
    meta = sample["meta"]
    out_name = f"pred_{meta['date']}_t{meta['time_idx']}_s{meta['canonical_step_idx']}_m{meta['member_idx']}.nc"
    out_path = os.path.join(args.output_dir, out_name)
    save_prediction(prediction[0, 0].cpu().numpy(), out_path)
    print(f">> Prediction saved to {out_path}")
    print(f">> Sample metadata: {meta}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained FiLM downscaling GAN")
    parser.add_argument("--architecture", type=str, default="unet", choices=["residual", "unet"],
                         help="Default: unet")
    parser.add_argument("--checkpoint", type=str, default=None, help="Required unless --dry_run")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config (recommended)")
    parser.add_argument("--data_dir", type=str, default=None, help="Legacy: single GRIB file. Ignored if --config given.")
    parser.add_argument("--region", type=str, default=None,
                         help="Override the config's region without editing the YAML, e.g. "
                              "--region ghana / --region west_africa / --region africa.")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        if args.checkpoint is None:
            parser.error("--checkpoint is required unless --dry_run is set")
        if args.config is None and args.data_dir is None:
            parser.error("--config or --data_dir is required unless --dry_run is set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dry_run:
        run_dry_run(args, device)
    elif args.config is not None:
        run_config_inference(args, device)
    else:
        run_grib_inference(args, device)


if __name__ == "__main__":
    main()
