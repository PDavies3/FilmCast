# FiLM-GAN Downscaling

A FiLM-conditioned, resolution-adaptive GAN for statistical downscaling of
ECMWF S2S subseasonal forecasts to high-resolution IMERG precipitation
fields, for West African rainfall applications.

Verified working on CPU: `pytest tests/ -v` -> 16 passed.

## Why this repo exists

Two generator architectures, both sharing the same design goals:

1. **Resolution-adaptive**: output resolution is read at forward-time from
   the static context tensor's shape, not hard-coded. Same trained weights
   work at any target resolution (tested up to and including deliberately
   awkward, non-power-of-2-divisible sizes) -- important since the real
   ECMWF (~1.5deg) -> IMERG (0.1deg) ratio is ~15x, not a clean power of 2.
2. **FiLM-conditioned on lead time and day-of-year**: rather than passing
   these as extra input channels, they modulate every layer via
   Feature-wise Linear Modulation, letting the network learn genuinely
   different behavior at different lead times (e.g. trust the input more
   at short lead times) and seasons (day-of-year is cyclically encoded via
   sin/cos so Dec 31 and Jan 1 are correctly treated as adjacent).

| Architecture | File | Family |
|---|---|---|
| `residual` | `models/film_adaptive_gan.py` | Residual blocks + single end-of-network upsample |
| `unet` | `models/film_unet_gan.py` | Encoder/decoder U-Net with skip connections, FiLM at every level |

## Structure
```
config_grib.py             # variable/path config for real ECMWF S2S GRIB2 data
inspect_grib.py             # inspect a GRIB file's variables/dims (all hypercubes)
list_grib_messages.py       # raw eccodes message inventory (ground truth, bypasses cfgrib)

models/
  __init__.py                # get_model(architecture, ...) factory: "residual" | "unet"
  film_adaptive_gan.py        # residual-block FiLM generator
  film_unet_gan.py            # U-Net FiLM generator
  discriminator.py            # shared PatchGAN discriminator

utils/
  conditioning.py             # builds [lead_time_norm, sin_doy, cos_doy]
  grib_dataset.py              # real-data loader: (date, time, step, member) indexing,
                                # valid_time-based IMERG alignment, canonical step-schedule
                                # alignment across variable groups with different resolutions

scripts/
  train.py                    # --architecture {residual,unet} --dry_run available
  infer.py                    # loads a checkpoint, runs inference, saves NetCDF

tests/
  conftest.py                  # synthetic fixtures (no files needed)
  test_models.py                # shape, resolution adaptivity, gradient flow
  test_conditioning.py          # cyclic day-of-year, lead-time normalization
```

## Setup
```bash
uv sync                    # or: pip install -r requirements.txt
pytest tests/ -v
```

## Training
```bash
# Dry run -- synthetic data, verifies the training loop mechanics with no real files
python scripts/train.py --architecture unet --dry_run --epochs 1

# Real run
export DOWNSCALING_DATA_ROOT="/path/to/your/grib/data"
python scripts/train.py --architecture unet --data_dir "$DOWNSCALING_DATA_ROOT" \
    --batch_size 8 --epochs 20 --checkpoint_dir ./checkpoints
```

## Inference
```bash
# Dry run
python scripts/infer.py --architecture unet --dry_run --output_dir ./predictions

# Real run
python scripts/infer.py --architecture unet --checkpoint ./checkpoints/unet_epoch20.pt \
    --data_dir "$DOWNSCALING_DATA_ROOT" --sample_index 0 --output_dir ./predictions
```

## Status / open items
- `config_grib.py`'s `SURFACE_VARIABLES` has some entries confirmed against
  real GRIB inspection and some still guessed -- see comments in the file.
- `STATIC_LAYERS_PATH` / `IMERG_FILENAME_TEMPLATE` are placeholders --
  point them at your real files.
- The discriminator is currently unconditioned (judges only the 1-channel
  precipitation field) -- FiLM-conditioning it too is a reasonable future
  upgrade, not yet implemented.
