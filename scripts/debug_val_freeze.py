"""
scripts/debug_val_freeze.py
------------------------------
One-off diagnostic: is validation loss frozen because (A) the validation
set is mostly dry and the metric is just insensitive at 4 decimals, or
(B) predictions genuinely aren't changing between checkpoints (a real
bug)? Run after you have at least two checkpoints to compare.

Usage:
    uv run python scripts/debug_val_freeze.py --checkpoint_dir ./checkpoints_debug \
        --checkpoints unet_epoch1.pt unet_epoch5.pt \
        --val_config Configs/quickstart_val.yaml
"""
import argparse
import numpy as np
import torch

from utils.config_loader import load_config
from utils.folder_dataset import ConfigurableDownscalingDataset
from models import ARCHITECTURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                         help="Two or more checkpoint filenames to compare, e.g. unet_epoch1.pt unet_epoch5.pt")
    parser.add_argument("--val_config", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="unet")
    args = parser.parse_args()

    config = load_config(args.val_config)
    ds = ConfigurableDownscalingDataset(config)

    zero_norm = -0.2963
    frac_dry = np.mean([float((ds[i]["target"] <= zero_norm + 1e-4).float().mean()) for i in range(len(ds))])
    print(f"Check 1 -- fraction of validation pixels at/near zero-precip: {frac_dry:.4f}")
    print(f"  (>0.90-0.95 suggests the validation period is mostly dry days)\n")

    ModelCls = ARCHITECTURES[args.architecture]
    netG = ModelCls(dynamic_in_channels=ds[0]["dynamic_input"].shape[0],
                     static_in_channels=ds[0]["static_input"].shape[0], cond_dim=3)

    sample = ds[0]
    dynamics = sample["dynamic_input"].unsqueeze(0)
    statics = sample["static_input"].unsqueeze(0)
    cond = sample["cond"].unsqueeze(0)

    print("Check 2 -- do raw predictions differ between checkpoints?")
    preds = {}
    for ckpt_name in args.checkpoints:
        ckpt = torch.load(f"{args.checkpoint_dir}/{ckpt_name}", map_location="cpu", weights_only=False)
        netG.load_state_dict(ckpt["generator_state_dict"])
        netG.eval()
        with torch.no_grad():
            out = netG(dynamics, statics, cond)
        preds[ckpt_name] = out.clone()
        print(f"  {ckpt_name}: mean={out.mean().item():.6f}  std={out.std().item():.6f}")

    names = list(preds.keys())
    if len(names) >= 2:
        diff = torch.abs(preds[names[0]] - preds[names[-1]]).mean().item()
        print(f"\n  Mean absolute difference, {names[0]} vs {names[-1]}: {diff:.8f}")
        if diff < 1e-6:
            print("  -> Predictions are IDENTICAL. This points to a real bug (Hypothesis B).")
        else:
            print("  -> Predictions DIFFER. Weights are updating (Hypothesis A likely -- validation metric is just insensitive).")


if __name__ == "__main__":
    main()
