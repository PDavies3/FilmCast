"""
scripts/validate_config.py
------------------------------
Beginner-friendly pre-flight check for a training config. Run this BEFORE
scripts/train.py to catch mistakes early with a plain-language explanation,
instead of discovering them partway through a training run.

Usage:
    python scripts/validate_config.py --config configs/my_config.yml

Checks, in order (stops at the first failure so you fix one thing at a
time rather than facing a wall of errors):
  1. The config file parses as valid YAML.
  2. The region is a known preset OR has all 4 required bounds.
  3. data_root exists on disk.
  4. Every input variable's folder exists under data_root.
  5. The dataset can actually be constructed (runs real discovery).
  6. Prints a plain-language summary: how many variables, what region,
     how many samples were found, and one example sample's shapes --
     so you can eyeball "does this look right?" before committing to a
     real (possibly hours-long) training run.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def validate(config_path):
    print(f"Checking {config_path} ...\n")

    # --- 1. YAML parses ---
    try:
        from utils.config_loader import load_config
        config = load_config(config_path)
    except Exception as e:
        print(f"FAILED at step 1/6: the config file isn't valid YAML, or a "
              f"referenced 'extends' file couldn't be found.\n  Details: {e}")
        return False
    print("[1/6] Config file parses correctly.")

    # --- 2. Region ---
    from utils.regions import resolve_region, REGIONS
    region_raw = config.get("region")
    try:
        region = resolve_region(region_raw)
    except ValueError as e:
        print(f"FAILED at step 2/6: {e}")
        return False
    required_keys = {"lat_min", "lat_max", "lon_min", "lon_max"}
    if not required_keys.issubset(region.keys()):
        print(f"FAILED at step 2/6: region is missing required bounds. "
              f"Needs all of {required_keys}, got {list(region.keys())}. "
              f"Either use a preset name ({list(REGIONS.keys())}) or "
              f"provide all four bounds explicitly.")
        return False
    print(f"[2/6] Region OK: {region_raw if isinstance(region_raw, str) else 'custom bounds'} "
          f"-> lat [{region['lat_min']}, {region['lat_max']}], lon [{region['lon_min']}, {region['lon_max']}]")

    # --- 3. data_root exists ---
    data_root = os.path.expanduser(config.get("data_root", ""))
    if not data_root:
        print("FAILED at step 3/6: config is missing 'data_root'.")
        return False
    if not os.path.isdir(data_root):
        print(f"FAILED at step 3/6: data_root '{data_root}' doesn't exist or "
              f"isn't a directory. If this uses !env \"${{SOME_VAR}}\", "
              f"check that environment variable is actually set: "
              f"echo ${{SOME_VAR}}")
        return False
    print(f"[3/6] data_root exists: {data_root}")

    # --- 4. Every input variable's folder exists ---
    inputs = config.get("inputs", [])
    if not inputs:
        print("FAILED at step 4/6: config has no 'inputs' -- need at least one predictor variable.")
        return False
    missing = []
    for spec in inputs:
        var_dir = os.path.join(data_root, spec["path"])
        if not os.path.isdir(var_dir):
            missing.append((spec["name"], var_dir))
    if missing:
        print(f"FAILED at step 4/6: {len(missing)} input variable folder(s) don't exist:")
        for name, path in missing:
            print(f"    '{name}' -> expected folder at {path}")
        return False
    print(f"[4/6] All {len(inputs)} input variable folder(s) exist: {[s['name'] for s in inputs]}")

    # --- 5. Dataset actually constructs (real discovery) ---
    try:
        from utils.folder_dataset import ConfigurableDownscalingDataset
        dataset = ConfigurableDownscalingDataset(config)
    except Exception as e:
        print(f"FAILED at step 5/6: dataset construction failed.\n  Details: {e}")
        return False
    print(f"[5/6] Dataset constructed successfully -- {len(dataset)} samples discovered.")

    # --- 6. Plain-language summary + one real sample ---
    try:
        sample = dataset[0]
    except Exception as e:
        print(f"FAILED at step 6/6: dataset built, but loading an actual "
              f"sample failed.\n  Details: {e}")
        return False

    print("[6/6] Loaded one real sample successfully.\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Input variables ({sample['dynamic_input'].shape[0]} channels total):")
    for spec in inputs:
        n_ch = len(spec.get("levels", [None]))
        print(f"    - {spec['name']}" + (f" ({n_ch} levels)" if n_ch > 1 else ""))
    print(f"  Target: {config['target']['name']}")
    print(f"  Additional data: {config.get('additional_data', [])}")
    print(f"  Total samples available: {len(dataset)}")
    print(f"  dynamic_input shape (this sample): {tuple(sample['dynamic_input'].shape)}")
    print(f"  static_input shape (this sample):  {tuple(sample['static_input'].shape)}")
    print(f"  target shape (this sample):        {tuple(sample['target'].shape)}")
    print(f"  Example sample metadata: {sample['meta']}")
    print()
    print("Looks good -- this config is ready for scripts/train.py")
    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-flight check a training config before running scripts/train.py")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    ok = validate(args.config)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
